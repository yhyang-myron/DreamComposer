from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
from ldm.util import instantiate_from_config
from datasets import load_dataset
import pytorch_lightning as pl
import copy
import csv
import cv2
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os, sys
import webdataset as wds
import math
from torch.utils.data.distributed import DistributedSampler
from ldm.modules.camera import T_blender_to_pinhole, T_to_pose, pose_opengl_to_opencv

from einops import rearrange


class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

        if 'image_transforms' in dataset_config:
            image_transforms = [torchvision.transforms.Resize(dataset_config.image_transforms.size)]
        else:
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)


    def train_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=False, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)
        # return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=True, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)
        # return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=self.validation),\
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='.objaverse/hf-objaverse-v1/views',
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=12,
        validation=False
        ) -> None:

        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        # with open(os.path.join(root_dir, 'valid_paths.json')) as f:
        #     self.paths = json.load(f)
        
        dir_paths = []
        first_level_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
        for first_dir in first_level_dirs:
            if not os.path.isdir(first_dir):
                continue
            for second_dir in os.listdir(first_dir):
                path = os.path.join(first_dir, second_dir)
                dir_paths.append(path)
        self.paths = dir_paths
            
        total_objects = len(self.paths)
        if validation:
            # self.paths = self.paths[:44]
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):] # used last 1% as validation
        else:
            # self.paths = self.paths[:44]
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)] # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms
        # print(self.paths)
        # exit(0)

    def __len__(self):
        return len(self.paths)
    
    def expand_tensors(self, tensors, dims):
        target = torch.zeros(dims)
        for i, tensor in enumerate(tensors):
            target[..., i] = tensor
        
        return target
        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T_org(self, target_pose, cond_pose):

        cond_pose = np.array(cond_pose)
        T_target = target_pose[:3, -1]

        T_cond = cond_pose[:3, -1]

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])

        # print(d_azimuth)
        # print((d_T[2] + 1) / 2)
        # print(torch.tensor(d_azimuth.item()) / (2 * math.pi))

        return d_T
    
    def get_T(self, target_pose, cond_pose_list):
        T_list = []
        lam_list = []
        for cond_pose in cond_pose_list:
            d_T = self.get_T_org(target_pose, cond_pose)
            T_list.append(d_T)
            lam = (d_T[2] + 1) / 2
            lam_list.append(lam)
        return T_list, lam_list

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        # print(path)
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img
    
    def cal_cam_pose(self, cond_pose_list):
        cond_pose_list_new = []
        for pose in cond_pose_list:
            pose = pose_opengl_to_opencv(pose)
            pose = pose.reshape(-1)
            intrinsics = np.array([1.1, 0, 0.5, 0, 1.1, 0.5, 0, 0, 1])
            camera_para = np.concatenate((pose, intrinsics))
            camera_para = torch.tensor(camera_para)
            cond_pose_list_new.append(camera_para)
        return cond_pose_list_new

    def __getitem__(self, index):
        # print(index)

        data = {}
        total_view = 72
        if random.random() > 0.5:
            index_target, index_cond_1, index_cond_3 = random.sample(range(36, total_view), 3) # without replacement
            index_cond_2 = ((index_cond_1 - 36) + 18) % 36 + 36
            if index_cond_3 == index_cond_2:
                index_cond_3 = (index_cond_2 - 36 + 9) % 36 + 36
            if random.random() > 0.5:
                index_target -= 36
        else:
            index_target, index_cond_1, index_cond_3 = random.sample(range(0, 36), 3) # without replacement
            index_cond_2 = (index_cond_1 + 18) % 36
            if index_cond_3 == index_cond_2:
                index_cond_3 = (index_cond_2 + 9) % 36
            if random.random() > 0.5:
                index_target += 36

        index_cond_list = [index_cond_1, index_cond_2, index_cond_3]
        valid_num = 3
        
        filename = self.paths[index]
        with open(os.path.join(filename, 'meta.json'), 'r') as file:
            meta_data = json.load(file)

        # print(self.paths[index])

        if self.return_paths:
            data["path"] = str(filename)
        
        color = [1., 1., 1., 1.]

        cond_im_list = []
        cond_pose_list = []
        target_im = self.process_im(self.load_im(os.path.join(filename, 'render_%04d.png' % index_target), color))
        target_pose = np.array(meta_data["locations"][index_target]["transform_matrix"])
        for index_cond in index_cond_list:
            cond_im_list.append(self.process_im(self.load_im(os.path.join(filename, 'render_%04d.png' % index_cond), color)))
            cond_pose_list.append(np.array(meta_data["locations"][index_cond]["transform_matrix"]))

        T_list, lam_list = self.get_T(target_pose, cond_pose_list)
        data["Ts"] = self.expand_tensors(T_list, [4,3])
        data["lam_org"] = self.expand_tensors(lam_list, [1,3])

        cond_pose_list = self.cal_cam_pose(cond_pose_list)
        data["con_cameras"] = self.expand_tensors(cond_pose_list, [25, 3])
        data["image_target"] = target_im
        image_conds = self.expand_tensors(cond_im_list, [256, 256, 3, 3])
        data["image_conds"] = image_conds
        data["valid_num"] = torch.tensor(valid_num)

        target_pose = pose_opengl_to_opencv(target_pose)
        target_pose = torch.tensor(target_pose).reshape(-1)
        intrinsics = torch.tensor([1.1, 0, 0.5, 0, 1.1, 0.5, 0, 0, 1])
        camera_para = torch.cat([target_pose, intrinsics])
        data["camera"] = camera_para


        if self.postprocess is not None:
            data = self.postprocess(data)
        
        # print(data.keys())
        # exit(0)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)
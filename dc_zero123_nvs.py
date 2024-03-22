import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.io import imsave
from os.path import join
import os
from PIL import Image
import math

from ldm.models.diffusion.DC_zero123 import DCZeroDiffusion
from ldm.util import instantiate_from_config, add_margin
from ldm.modules.camera import T_blender_to_pinhole, T_to_pose, pose_opengl_to_opencv, pose_opencv_to_opengl, spherical_to_Pose_towards_origin

def cartesian_to_spherical(xyz):
    # ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    z = np.sqrt(xy + xyz[:,2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azimuth = np.arctan2(xyz[:,1], xyz[:,0])
    return np.array([theta, azimuth, z])

def get_T_org(target_pose, cond_pose):

    cond_pose = np.array(cond_pose)
    T_target = target_pose[:3, -1]

    T_cond = cond_pose[:3, -1]

    theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(T_cond[None, :])
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target[None, :])
    
    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
    d_z = z_target - z_cond
    d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])

    return d_T

def get_T(target_pose, cond_pose_list):
    T_list = []
    lam_list = []
    for cond_pose in cond_pose_list:
        d_T = get_T_org(target_pose, cond_pose)
        T_list.append(d_T)
        lam = (d_T[2] + 1) / 2
        lam_list.append(lam)
    return T_list, lam_list

def cal_cam_pose(cond_pose_list):
    cond_pose_list_new = []
    for pose in cond_pose_list:
        pose = pose_opengl_to_opencv(pose)
        pose = pose.reshape(-1)
        intrinsics = np.array([1.1, 0, 0.5, 0, 1.1, 0.5, 0, 0, 1])
        camera_para = np.concatenate((pose, intrinsics))
        camera_para = torch.tensor(camera_para)
        cond_pose_list_new.append(camera_para)
    return cond_pose_list_new

def prepare_inputs_control(input_path_list, input_azimuth_list, target_azimuth, camera_distance, elevation, crop_size=-1, image_size=256):
    # data = {}
    s = len(input_path_list)
    input_image_list = []
    for input_path in input_path_list:
        image_input = Image.open(input_path)

        if crop_size!=-1:
            alpha_np = np.asarray(image_input)[:, :, 3]
            coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
            min_x, min_y = np.min(coords, 0)
            max_x, max_y = np.max(coords, 0)
            ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
            h, w = ref_img_.height, ref_img_.width
            scale = crop_size / max(h, w)
            h_, w_ = int(scale * h), int(scale * w)
            ref_img_ = ref_img_.resize((w_, h_), resample=Image.BICUBIC)
            image_input = add_margin(ref_img_, size=image_size)
        else:
            image_input = add_margin(image_input, size=max(image_input.height, image_input.width))
            image_input = image_input.resize((image_size, image_size), resample=Image.BICUBIC)
        
        image_input = np.asarray(image_input)
        image_input = image_input.astype(np.float32) / 255.0
        ref_mask = image_input[:, :, 3:]
        image_input[:, :, :3] = image_input[:, :, :3] * ref_mask + 1 - ref_mask  # white background

        image_input = image_input[:, :, :3] * 2.0 - 1.0
        image_input = torch.from_numpy(image_input.astype(np.float32))
        input_image_list.append(image_input)
    input_images = torch.stack(input_image_list, -1)  # (h, w, c, s)

    input_pose_list = []
    # poses = np.load(f'meta_info/DCsync_camera_pose.npy')
    # poses = []
    # for i in range(16):
    #     poses.append(spherical_to_Pose_towards_origin(camera_distance, 0.5 * np.pi - np.deg2rad(30), np.deg2rad(i * 22.5)))
    # poses = np.array(poses)
    # poses.append(spherical_to_Pose_towards_origin(camera_distance, 0.5 * np.pi - np.deg2rad(30), np.deg2rad(target_azimuth)))

    for input_azimuth in input_azimuth_list:
        cond_pose_opencv = spherical_to_Pose_towards_origin(camera_distance, 0.5 * np.pi - np.deg2rad(elevation), np.deg2rad(input_azimuth))
        cond_pose_opengl = pose_opencv_to_opengl(cond_pose_opencv)
        input_pose_list.append(cond_pose_opengl)
    valid_num = s

    target_pose = spherical_to_Pose_towards_origin(camera_distance, 0.5 * np.pi - np.deg2rad(30), np.deg2rad(target_azimuth))
    target_pose_opengl = pose_opencv_to_opengl(target_pose)
    T_list, lam_list = get_T(target_pose_opengl, input_pose_list)
    Ts = torch.stack(T_list, -1)
    lam_org = torch.stack(lam_list, -1)
    camera_pose = torch.tensor(target_pose)
    camera_pose = camera_pose.reshape(-1)
    intrinsics = torch.tensor([1.1, 0, 0.5, 0, 1.1, 0.5, 0, 0, 1])
    camera_pose = torch.cat([camera_pose, intrinsics])
    camera_para = camera_pose


    input_pose_list = cal_cam_pose(input_pose_list)
    con_cameras = torch.stack(input_pose_list, -1)  # (25, s)
    valid_num_data = torch.tensor(valid_num)
    elevation_input = torch.from_numpy(np.asarray(np.deg2rad(elevation), np.float32))

    return {"input_image": input_images[..., 0], "input_elevation": elevation_input, 
                "image_conds": input_images, "con_cameras": con_cameras, "Ts": Ts, "lam_org": lam_org, "camera": camera_para, "valid_num": valid_num_data}

def load_model(cfg,ckpt,strict=True):
    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.model)
    print(f'loading model from {ckpt} ...')
    ckpt = torch.load(ckpt,map_location='cpu')
    model.load_state_dict(ckpt['state_dict'],strict=strict)
    model = model.cuda().eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',type=str, default='configs/DC_zero123.yaml')
    parser.add_argument('--ckpt',type=str, default='ckpt/DC_zero123.ckpt')
    parser.add_argument('--output', type=str, default='output_imgs/dc_zero123/alarm')
    parser.add_argument('--elevation', type=float, default=30)
    parser.add_argument('--target_azim', type=float, default=45)

    parser.add_argument('--sample_num', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=-1)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=6033)

    parser.add_argument('--sample_steps', type=int, default=200)
    flags = parser.parse_args()

    torch.random.manual_seed(flags.seed)
    np.random.seed(flags.seed)

    model = load_model(flags.cfg, flags.ckpt, strict=True)
    assert isinstance(model, DCZeroDiffusion)
    Path(f'{flags.output}').mkdir(exist_ok=True, parents=True)

    # prepare data
    input_path_list = ['images/alarm/0.webp',
                        'images/alarm/90.webp']
    input_azimuth_list = [0., 90.]
    assert len(input_path_list) > 1
    assert len(input_path_list) == len(input_azimuth_list)

    elevation = flags.elevation
    crop_size = flags.crop_size
    target_azim = flags.target_azim
    # You may also adjust the camera_distance for better results
    if crop_size == -1:
        camera_distance = 1.8
    else:
        camera_distance = 1.8 + ((200 - crop_size) / 100.)
    camera_distance = max(1.6, min(camera_distance, 2.2))
    image_size = 256
    data = prepare_inputs_control(input_path_list, input_azimuth_list, target_azim, camera_distance, elevation, crop_size, image_size)

    for k, v in data.items():
        data[k] = v.unsqueeze(0).cuda()
        data[k] = torch.repeat_interleave(data[k], flags.sample_num, dim=0)
        
    # sampler = DCSyncDDIMSampler(model, flags.sample_steps)
    x_sample = model.sample(data, flags.cfg_scale, flags.sample_steps)

    B, _, H, W = x_sample.shape
    x_sample = (torch.clamp(x_sample,max=1.0,min=-1.0) + 1) * 0.5
    x_sample = x_sample.permute(0,2,3,1).cpu().numpy() * 255
    x_sample = x_sample.astype(np.uint8)

    output_path = join(flags.output, f'seed{flags.seed}_elev{int(elevation)}_cs{int(crop_size)}_cam{camera_distance:.1f}_cfg{flags.cfg_scale:.1f}_step{int(flags.sample_steps)}')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for bi in range(B):
        imsave(join(output_path, f'{bi}.png'), x_sample[bi])
        # imsave(join(output_path, f'{bi}_sync.png'), np.concatenate([x_sample_org[bi,ni] for ni in range(N)], 1))
    print(output_path, 'done!!')


if __name__ == '__main__':
    main()

from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.io import imsave
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from einops import rearrange

from ldm.base_utils import read_pickle, concat_images_list
from ldm.util import instantiate_from_config

from ldm.models.diffusion.sync_dreamer import SyncMultiviewDiffusion, SyncDDIMSampler, repeat_to_batch
from ldm.models.planeencoder import UNetEncoderWeightedRecon

from PIL import Image
import os
import copy


def vis_latent(pred, target):
    save_path = 'latent_vis'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for bi in range(pred.shape[0]):
        if bi == 16:
            break
        latent_pred = pred[bi]
        # if bi == 0 or bi == 1:
        #     print(torch.unique(latent_pred, return_counts=True))
        latent_gt = target[bi]
        latent_pred = ((latent_pred + 1.0) / 2.0).detach().cpu().numpy() * 255
        latent_pred = latent_pred.astype(np.uint8)
        latent_gt = ((latent_gt + 1.0) / 2.0).detach().cpu().numpy() * 255
        latent_gt = latent_gt.astype(np.uint8)
        Image.fromarray(latent_pred).save(os.path.join(save_path, f'B{bi}_latent_pred.png'))
        # Image.fromarray(latent_gt).save(os.path.join(save_path, f'B{bi}_latent_gt.png'))
        

class DCUNetWrapper(nn.Module):

    def __init__(self, diff_model_config, drop_conditions=False, drop_scheme='default', 
                use_zero_123=True, control_stage_config=None):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.drop_conditions = drop_conditions
        self.drop_scheme=drop_scheme
        self.use_zero_123 = use_zero_123

        self.control_model = instantiate_from_config(control_stage_config)
        self.control_scales = [1.0] * 13
        self.only_mid_control = False


    def drop(self, cond, mask):
        shape = cond.shape
        B = shape[0]
        cond = mask.view(B,*[1 for _ in range(len(shape)-1)]) * cond
        return cond

    def get_trainable_parameters(self):
        return self.diffusion_model.get_trainable_parameters()

    def get_drop_scheme(self, B, device):
        if self.drop_scheme=='default':
            random = torch.rand(B, dtype=torch.float32, device=device)
            drop_clip = (random > 0.15) & (random <= 0.2)
            drop_volume = (random > 0.1) & (random <= 0.15)
            drop_concat = (random > 0.05) & (random <= 0.1)
            drop_all = random <= 0.05
        else:
            raise NotImplementedError
        return drop_clip, drop_volume, drop_concat, drop_all

    def predict_with_unconditional_scale(self, x, t, cond, volume_feats, unconditional_scale):
        clip_embed = cond["clip_embed"]
        x_concat = cond["x_concat"]

        x_ = torch.cat([x] * 2, 0)
        t_ = torch.cat([t] * 2, 0)
        clip_embed_ = torch.cat([clip_embed, torch.zeros_like(clip_embed)], 0)

        v_ = {}
        for k, v in volume_feats.items():
            v_[k] = torch.cat([v, torch.zeros_like(v)], 0)

        x_concat_ = torch.cat([x_concat, torch.zeros_like(x_concat)], 0)
        if self.use_zero_123:
            # zero123 does not multiply this when encoding, maybe a bug for zero123
            first_stage_scale_factor = 0.18215
            x_concat_[:, :4] = x_concat_[:, :4] / first_stage_scale_factor
        x_ = torch.cat([x_, x_concat_], 1)

        hint = cond['hint']
        hint_ = torch.cat([hint, torch.zeros_like(hint)], 0)
        # hint_ = torch.cat([hint, hint], 0)

        if 'single' in cond:
            control = None
        else:
            control = self.control_model(x=x_, hint=hint_, timesteps=t_, context=clip_embed_)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
        s, s_uc = self.diffusion_model(x_, t_, clip_embed_, source_dict=v_, control=control).chunk(2)
        s = s_uc + unconditional_scale * (s - s_uc)
        return s


class DCSyncDiffusion(SyncMultiviewDiffusion):
    def __init__(self, lift3d_config, control_stage_config, recon_weight=0.1, only_mid_control=False, drop_conditions=False, 
                drop_scheme='default', unet_config=None, finetune_volume=False, recon_loss=True, *args, **kwargs):
        super().__init__(unet_config=unet_config, drop_conditions=drop_conditions, drop_scheme=drop_scheme, *args, **kwargs)

        self.finetune_volume = finetune_volume
        self.recon_loss = recon_loss

        self.model = DCUNetWrapper(unet_config, drop_conditions=drop_conditions, drop_scheme=drop_scheme, control_stage_config=control_stage_config)
        self._init_ldm()
        self.Lift3D = UNetEncoderWeightedRecon(unet_config=lift3d_config)
        self.recon_weight = recon_weight

        latent_size = self.image_size//8
        if self.sample_type=='ddim':
            self.sampler = DCSyncDDIMSampler(self, self.sample_steps , "uniform", 1.0, latent_size=latent_size)
        else:
            raise NotImplementedError
    
    @torch.no_grad()
    def _init_ldm(self):
        for param in self.model.diffusion_model.input_blocks.parameters():
            param.requires_grad = False
        for param in self.model.diffusion_model.middle_block.parameters():
            param.requires_grad = False
        for param in self.model.diffusion_model.time_embed.parameters():
            param.requires_grad = False
    
    def configure_optimizers(self):
        lr = self.learning_rate
        print(f'setting learning rate to {lr:.4f} ...')

        params = []
        if self.finetune_volume:
            params.append({"params": self.time_embed.parameters(), "lr": lr},)
            params.append({"params": self.spatial_volume.parameters(), "lr": lr},)
            params.append({"params": self.model.get_trainable_parameters(), "lr": lr},)

        params.append({"params": self.model.control_model.parameters(), "lr": 10 * lr},)
        params.append({"params": self.Lift3D.parameters(), "lr": lr},)
        opt = torch.optim.AdamW(params, lr=lr)

        scheduler = instantiate_from_config(self.scheduler_config)
        print("Setting up LambdaLR scheduler...")
        scheduler = [{'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule), 'interval': 'step', 'frequency': 1}]

        return [opt], scheduler
    
    def prepare(self, batch):
        # encode target
        if 'target_image' in batch:
            image_target = batch['target_image'].permute(0, 1, 4, 2, 3) # b,n,3,h,w
            N = image_target.shape[1]
            x = [self.encode_first_stage(image_target[:,ni], True) for ni in range(N)]
            x = torch.stack(x, 1) # b,n,4,h//8,w//8
        else:
            x = None

        image_input = batch['input_image'].permute(0, 3, 1, 2)
        elevation_input = batch['input_elevation']
        b, h, w, c, s = batch['image_conds'].shape
        image_inputs = rearrange(batch['image_conds'], 'b h w c s -> (b s) c h w')
        img_inputs_latent = self.encode_first_stage(image_inputs, sample=False)
        x_input = rearrange(img_inputs_latent, '(b s) c h w -> b s c h w', s=s)[:, 0, :, :, :]

        input_info = {'image': image_input, 'elevation': elevation_input, 'x': x_input, 'c_concat': img_inputs_latent, 'image_inputs': image_inputs}
        with torch.no_grad():
            clip_embed = self.clip_image_encoder.encode(image_input)
        return x, clip_embed, input_info

    def training_step(self, batch, batch_idx):

        B = batch['target_image'].shape[0]
        time_steps = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()

        x, clip_embed, input_info = self.prepare(batch)
        x_noisy, noise = self.add_noise(x, time_steps)  # B,N,4,H,W
        N = self.view_num
        target_index = torch.randint(0, N, (B, 1), device=self.device).long() # B, 1
        B_range = torch.arange(B)[:,None]
        cond = {}

        # Triplane
        tri_dict = {}
        Ts = rearrange(batch["Ts"], 'b c s n -> b n c s')[B_range,target_index][:,0]
        Ts = rearrange(Ts, 'b c s -> (b s) c')
        tri_dict["Ts"] = Ts

        first_stage_scale_factor = 0.18215
        c_concat = input_info["c_concat"] * 1.0
        c_concat[:, :4] = c_concat[:, :4] / first_stage_scale_factor
        tri_dict["c_concat"] = c_concat  # (b*s, c, h, w)

        camera = rearrange(batch["camera"], 'b c n -> b n c')[B_range,target_index][:,0]
        tri_dict["camera"] = camera
        tri_dict["con_cameras"] = batch["con_cameras"]  # (b c s)
        tri_dict["valid_num"] = batch["valid_num"]
        lam_org = rearrange(batch["lam_org"], 'b s n -> b n s')[B_range,target_index][:,0]
        lam_org = rearrange(lam_org, 'b s -> (b s)')
        tri_dict["lam_org"] = lam_org
        output_dict = self.Lift3D(tri_dict)
        pred = output_dict["feat"]  # (b c h w)
        cond["hint"] = pred

        # Volume
        v_embed = self.get_viewpoint_embedding(B, input_info['elevation']) # N,v_dim
        t_embed = self.embed_time(time_steps)
        spatial_volume = self.spatial_volume.construct_spatial_volume(x_noisy, t_embed, v_embed, self.poses, self.Ks)

        clip_embed, volume_feats, x_concat = self.get_target_view_feats(input_info['x'], spatial_volume, clip_embed, t_embed, v_embed, target_index)

        x_noisy_ = x_noisy[B_range,target_index][:,0] # B,4,H,W
        cond["clip_embed"] = clip_embed
        cond["x_concat"] = x_concat
        noise_predict = self.model(x_noisy_, time_steps, cond, volume_feats, is_train=True) # B,4,H,W

        noise_target = noise[B_range,target_index][:,0] # B,4,H,W
        # loss simple for diffusion
        loss_simple = torch.nn.functional.mse_loss(noise_target, noise_predict, reduction='none')
        loss = loss_simple.mean()
        self.log('loss_sim', loss_simple.mean(), prog_bar=True, logger=True, on_step=True, on_epoch=True, rank_zero_only=True)
        
        if self.recon_loss:
            # loss for reconstruction
            latent_gt = x[B_range,target_index][:,0]
            loss_recon = torch.nn.functional.mse_loss(latent_gt, pred, reduction='none')
            loss_recon = loss_recon.mean()
            self.log('loss_recon', loss_recon.mean(), prog_bar=True, logger=True, on_step=True, on_epoch=True, rank_zero_only=True)

            loss = loss + self.recon_weight * loss_recon

        # log others
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        self.log("step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if (batch_idx==0 or batch_idx==1) and self.global_rank==0:
            self.eval()
            permuted_indices = torch.randperm(40)
            selected_index = permuted_indices[:self.output_num]
            batch_ = {}
            for k, v in batch.items(): 
                batch_[k] = v[selected_index]
            self.log_images(batch_, split="val", batch_idx=batch_idx, batch_view_num=self.batch_view_num)
    
    @torch.no_grad()
    def log_images(self, batch, split, return_inter_results=False, inter_interval=50, batch_view_num=1, inter_view_interval=2, batch_idx=0):     
        # B = batch['target_image'].shape[0]
        # time_steps = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()

        x, clip_embed, input_info = self.prepare(batch)
        cond = {}

        # Triplane
        tri_dict = {}
        b, _, s, n = batch["Ts"].shape
        Ts = rearrange(batch["Ts"], 'b c s n -> (b n s) c')
        tri_dict["Ts"] = Ts

        first_stage_scale_factor = 0.18215
        c_concat = input_info["c_concat"] * 1.0
        c_concat[:, :4] = c_concat[:, :4] / first_stage_scale_factor

        tri_dict["c_concat"] = rearrange(c_concat.unsqueeze(-1).repeat(1, 1, 1, 1, n), '(b s) c h w n -> (b n s) c h w', s=s)  # (b*n*s, c, h, w)
        camera = rearrange(batch["camera"], 'b c n -> (b n) c')
        tri_dict["camera"] = camera
        tri_dict["con_cameras"] = rearrange(batch["con_cameras"].unsqueeze(-1).repeat(1, 1, 1, n), 'b c s n -> (b n) c s')  # (b*n c s)
        tri_dict["valid_num"] = batch["valid_num"].repeat(n)
        lam_org = rearrange(batch["lam_org"], 'b s n -> (b n s)')
        tri_dict["lam_org"] = lam_org
        output_dict = self.Lift3D(tri_dict)
        pred = output_dict["feat"]  # (b*n)

        cond["hint"] = rearrange(pred, '(b n) c h w -> b n c h w', n=n)
        cond["clip_embed"] = clip_embed

        output_dir = Path(self.image_dir) / 'images' / split
        output_dir.mkdir(exist_ok=True, parents=True)

        x_sample, inter = self.sampler.sample(input_info, cond, unconditional_scale=self.cfg_scale, log_every_t=inter_interval, batch_view_num=batch_view_num)
        N = x_sample.shape[1]
        x_sample = torch.stack([self.decode_first_stage(x_sample[:, ni]) for ni in range(N)], 1)
        if return_inter_results:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            inter = torch.stack(inter['x_inter'], 2) # # B,N,T,C,H,W
            B,N,T,C,H,W = inter.shape
            inter_results = []
            for ni in tqdm(range(0, N, inter_view_interval)):
                inter_results_ = []
                for ti in range(T):
                    inter_results_.append(self.decode_first_stage(inter[:, ni, ti]))
                inter_results.append(torch.stack(inter_results_, 1)) # B,T,3,H,W
            inter_results = torch.stack(inter_results,1) # B,N,T,3,H,W
        
        process = lambda x: ((torch.clip(x, min=-1, max=1).cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
        B = x_sample.shape[0]
        N = x_sample.shape[1]
        image_cond = []
        image_inputs = batch["image_conds"]
        # print(process(batch['input_image'][0]).shape, process(image_inputs[0][..., 1]).shape)

        for bi in range(B):
            if s == 3:
                img_pr_ = concat_images_list(process(batch['input_image'][bi]), process(image_inputs[bi][..., 1]), process(image_inputs[bi][..., 2]), *[process(x_sample[bi, ni].permute(1, 2, 0)) for ni in range(N)])
            elif s == 2:
                img_pr_ = concat_images_list(process(batch['input_image'][bi]), process(image_inputs[bi][..., 1]), *[process(x_sample[bi, ni].permute(1, 2, 0)) for ni in range(N)])
            image_cond.append(img_pr_)
        output_imgs = concat_images_list(*image_cond, vert=True)
        imsave(str(output_dir/f'{self.global_step}_{batch_idx}_ours.jpg'), output_imgs)

        # B = 8
        # N = 16

        triplane_pred = []
        img_pred = self.decode_first_stage(pred)
        img_pred = rearrange(img_pred, '(b n) c h w -> b n c h w', n=n)
        for bi in range(B):
            tri_pr_ = concat_images_list(*[process(img_pred[bi, ni].permute(1, 2, 0)) for ni in range(N)])
            triplane_pred.append(tri_pr_)
        imsave(str(output_dir/f'{self.global_step}_{batch_idx}_latent_pred.jpg'), concat_images_list(*triplane_pred, vert=True))

        img_gt = []
        target_image = batch["target_image"]
        for bi in range(B):
            img_gt_ = concat_images_list(*[process(target_image[bi, ni]) for ni in range(N)])
            img_gt.append(img_gt_)
        imsave(str(output_dir/f'{self.global_step}_{batch_idx}_img_gt.jpg'), concat_images_list(*img_gt, vert=True))

        # org_results
        cond["zero123"] = torch.zeros(1)
        x_sample, inter = self.sampler.sample(input_info, cond, unconditional_scale=self.cfg_scale, log_every_t=inter_interval, batch_view_num=batch_view_num)
        N = x_sample.shape[1]
        x_sample = torch.stack([self.decode_first_stage(x_sample[:, ni]) for ni in range(N)], 1)
        if return_inter_results:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            inter = torch.stack(inter['x_inter'], 2) # # B,N,T,C,H,W
            B,N,T,C,H,W = inter.shape
            inter_results = []
            for ni in tqdm(range(0, N, inter_view_interval)):
                inter_results_ = []
                for ti in range(T):
                    inter_results_.append(self.decode_first_stage(inter[:, ni, ti]))
                inter_results.append(torch.stack(inter_results_, 1)) # B,T,3,H,W
            inter_results = torch.stack(inter_results,1) # B,N,T,3,H,W
        
        process = lambda x: ((torch.clip(x, min=-1, max=1).cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
        B = x_sample.shape[0]
        N = x_sample.shape[1]
        image_cond = []
        for bi in range(B):
            img_pr_ = concat_images_list(process(batch['input_image'][bi]),*[process(x_sample[bi, ni].permute(1, 2, 0)) for ni in range(N)])
            image_cond.append(img_pr_)
        output_imgs_org = concat_images_list(*image_cond, vert=True)
        imsave(str(output_dir/f'{self.global_step}_{batch_idx}_syncdreamer.jpg'), output_imgs_org)
    
    @torch.no_grad()
    def sample(self, sampler, batch, cfg_scale, batch_view_num, return_inter_results=False, inter_interval=50, inter_view_interval=2):
        x, clip_embed, input_info = self.prepare(batch)
        cond = {}

        # Triplane
        tri_dict = {}
        b, _, s, n = batch["Ts"].shape
        Ts = rearrange(batch["Ts"], 'b c s n -> (b n s) c')
        tri_dict["Ts"] = Ts

        first_stage_scale_factor = 0.18215
        c_concat = input_info["c_concat"] * 1.0
        c_concat[:, :4] = c_concat[:, :4] / first_stage_scale_factor

        tri_dict["c_concat"] = rearrange(c_concat.unsqueeze(-1).repeat(1, 1, 1, 1, n), '(b s) c h w n -> (b n s) c h w', s=s)  # (b*n*s, c, h, w)
        camera = rearrange(batch["camera"], 'b c n -> (b n) c')
        tri_dict["camera"] = camera
        tri_dict["con_cameras"] = rearrange(batch["con_cameras"].unsqueeze(-1).repeat(1, 1, 1, n), 'b c s n -> (b n) c s')  # (b*n c s)
        tri_dict["valid_num"] = batch["valid_num"].repeat(n)
        lam_org = rearrange(batch["lam_org"], 'b s n -> (b n s)')
        tri_dict["lam_org"] = lam_org
        output_dict = self.Lift3D(tri_dict)
        pred = output_dict["feat"]  # (b*n)

        cond["hint"] = rearrange(pred, '(b n) c h w -> b n c h w', n=n)
        cond["clip_embed"] = clip_embed

        # Visualize Latent Output
        # latent_pred = rearrange(self.decode_first_stage(pred), 'b c h w -> b h w c')
        # vis_latent(latent_pred, latent_pred)

        # results
        x_sample, inter = sampler.sample(input_info, cond, unconditional_scale=cfg_scale, log_every_t=inter_interval, batch_view_num=batch_view_num)
        N = x_sample.shape[1]
        x_sample = torch.stack([self.decode_first_stage(x_sample[:, ni]) for ni in range(N)], 1)

        return x_sample


class DCSyncDDIMSampler(SyncDDIMSampler):
    @torch.no_grad()
    def denoise_apply(self, x_target_noisy, input_info, cond, time_steps, index, unconditional_scale, batch_view_num=1, is_step0=False):
        """
        @param x_target_noisy:   B,N,4,H,W
        @param input_info:
        @param clip_embed:       B,M,768
        @param time_steps:       B,
        @param index:            int
        @param unconditional_scale:
        @param batch_view_num:   int
        @param is_step0:         bool
        @return:
        """
        clip_embed = cond["clip_embed"]

        x_input, elevation_input = input_info['x'], input_info['elevation']
        B, N, C, H, W = x_target_noisy.shape

        # construct source data
        v_embed = self.model.get_viewpoint_embedding(B, elevation_input) # B,N,v_dim
        t_embed = self.model.embed_time(time_steps)  # B,t_dim
        spatial_volume = self.model.spatial_volume.construct_spatial_volume(x_target_noisy, t_embed, v_embed, self.model.poses, self.model.Ks)

        # SyncDreamer + ControlDreamer
        e_t = []
        target_indices = torch.arange(N) # N
        B_range = torch.arange(B)[:,None]
        # print(batch_view_num)
        for ni in range(0, N, batch_view_num):
            x_target_noisy_ = x_target_noisy[:, ni:ni + batch_view_num]
            VN = x_target_noisy_.shape[1]
            x_target_noisy_ = x_target_noisy_.reshape(B*VN,C,H,W)

            time_steps_ = repeat_to_batch(time_steps, B, VN)
            target_indices_ = target_indices[ni:ni+batch_view_num].unsqueeze(0).repeat(B,1)
            clip_embed_, volume_feats_, x_concat_ = self.model.get_target_view_feats(x_input, spatial_volume, clip_embed, t_embed, v_embed, target_indices_)
            cond_ = {}
            cond_["clip_embed"] = clip_embed_
            cond_["x_concat"] = x_concat_
            hint_ = cond["hint"]
            hint_ = hint_[B_range,target_indices_].reshape(B*VN,C,H,W)
            cond_["hint"] = hint_
            if 'single' in cond:
                cond_["single"] = cond["single"]
            if unconditional_scale!=1.0:
                noise = self.model.model.predict_with_unconditional_scale(x_target_noisy_, time_steps_, cond_, volume_feats_, unconditional_scale)
            else:
                noise = self.model.model(x_target_noisy_, time_steps_, cond_, volume_feats_, is_train=False)
            e_t.append(noise.view(B,VN,4,H,W))

        e_t = torch.cat(e_t, 1)
        x_prev = self.denoise_apply_impl(x_target_noisy, index, e_t, is_step0)
    
        return x_prev

    @torch.no_grad()
    def sample(self, input_info, cond, unconditional_scale=1, log_every_t=50, batch_view_num=1):
        """
        @param input_info:      x, elevation
        @param clip_embed:      B,M,768
        @param unconditional_scale:
        @param log_every_t:
        @param batch_view_num:
        @return:
        """
        clip_embed = cond["clip_embed"]

        print(f"unconditional scale {unconditional_scale:.1f}")
        C, H, W = 4, self.latent_size, self.latent_size
        B = clip_embed.shape[0]
        N = self.model.view_num
        device = self.model.device
        x_target_noisy = torch.randn([B, N, C, H, W], device=device)

        timesteps = self.ddim_timesteps
        intermediates = {'x_inter': []}
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1 # index in ddim state
            time_steps = torch.full((B,), step, device=device, dtype=torch.long)
            x_target_noisy = self.denoise_apply(x_target_noisy, input_info, cond, time_steps, index, unconditional_scale, batch_view_num=batch_view_num, is_step0=index==0)
            # x_target_org_noisy = self.denoise_apply(x_target_org_noisy, input_info, cond_org, time_steps, index, unconditional_scale, batch_view_num=batch_view_num, is_step0=index==0)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(x_target_noisy)

        return x_target_noisy, intermediates
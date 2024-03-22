"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager, nullcontext
from functools import partial
import itertools
from tqdm import tqdm
from torchvision.utils import make_grid
from omegaconf import ListConfig

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config, repeat_interleave
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.attention import CrossAttention

from ldm.models.planeencoder import UNetEncoderWeightedRecon
from ldm.models.diffusion.ddpm import LatentDiffusion

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def disable_training_module(module: nn.Module):
    module = module.eval()
    module.train = disabled_train
    for para in module.parameters():
        para.requires_grad = False
    return module


class DCZeroDiffusion(LatentDiffusion):
    def __init__(self, lift3d_config, control_stage_config, only_mid_control=False, recon_weight=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.Lift3D = UNetEncoderWeightedRecon(unet_config=lift3d_config)
        self.control_model = instantiate_from_config(control_stage_config)
        self.recon_weight = recon_weight
        self.control_scales = [1.0] * 13
        self.only_mid_control = only_mid_control

        self.cc_projection = disable_training_module(self.cc_projection)
        self._init_ldm()
        # self.first_stage_model = disable_training_module(self.first_stage_model)  # already done in LatentDiffusion
        # self.cond_stage_model = disable_training_module(self.cond_stage_model)
    
    torch.no_grad()
    def _init_ldm(self):
        for param in self.model.diffusion_model.input_blocks.parameters():
            param.requires_grad = False
        for param in self.model.diffusion_model.middle_block.parameters():
            param.requires_grad = False
        for param in self.model.diffusion_model.time_embed.parameters():
            param.requires_grad = False
    
    def shared_step(self, batch, **kwargs):
        pred, target, cond = self.get_input(batch, 'image_target')
        loss_sim, loss_dict = self(target, cond)

        loss_recon = self.get_loss(pred, target, mean=True)
        loss = loss_sim + self.recon_weight * loss_recon

        prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{prefix}/loss_recon': loss_recon})
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict
    
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_img = torch.cat(cond['c_crossattn'], 1)
        c_concat = cond['c_concat']
        xc = torch.cat([x_noisy] + c_concat, dim=1)
        # print(xc.shape)
        # print(cond_img.shape)
        
        if 'zero123' in cond:
            control = None
        else:
            control = self.control_model(x=xc, hint=torch.cat(cond['hint'], 1), timesteps=t, context=cond_img)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
        eps = diffusion_model(x=xc, timesteps=t, context=cond_img, control=control, only_mid_control=self.only_mid_control)

        return eps
    
    @torch.no_grad()
    def get_input_DF(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, uncond=0.05):
        x = self.get_input_DF(batch, k)
        Ts = batch['Ts'].to(memory_format=torch.contiguous_format).float() # (bs, 4, 3)
        if bs is not None:
            x = x[:bs]
            Ts = Ts[:bs].to(self.device)
            # Tcons = Tcons[:bs].to(self.device)
            batch['Ts'] = Ts
            # batch['Tcons'] = Tcons
            # batch["depth_cons"] = batch["depth_cons"][:bs]
            batch["con_cameras"] = batch["con_cameras"][:bs]
            batch["lam_org"] = batch["lam_org"][:bs]
        T_1 = Ts[..., 0]
        Ts = rearrange(Ts, 'b c s -> (b s) c')
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        xcs = batch["image_conds"].float()
        if bs is not None:
            xcs = xcs[:bs]
        B, H, W, C, S = xcs.shape
        xcs = rearrange(xcs, 'b h w c s -> b c h w s')
        xc_1 = xcs[..., 0]
        xc_2 = xcs[..., 1]
        xc_3 = xcs[..., 2]
        xcs = rearrange(xcs, 'b c h w s -> (b s) c h w')
        latent_imgs = self.encode_first_stage((xcs.to(self.device))).mode().detach()
        batch["c_concat"] = latent_imgs

        if bs is not None:
            batch["camera"] = batch["camera"][:bs]
            batch["con_cameras"] = batch["con_cameras"]
            batch["valid_num"] = batch["valid_num"][:bs]
        
        cond = {}
        # random = torch.rand(x.size(0), device=x.device)
        # prompt_mask = rearrange(random < 2 * uncond, "n -> n 1 1")
        # input_mask = 1 - rearrange((random >= uncond).float() * (random < 3 * uncond).float(), "n -> n 1 1 1")
        # null_prompt = self.get_learned_conditioning([""])
        clip_emb_1 = self.get_learned_conditioning(xc_1).detach()
        # null_prompt = self.get_learned_conditioning([""]).detach()
        # c_crossattn_org = self.cc_projection(torch.cat([torch.where(prompt_mask, null_prompt, clip_emb_1), T_1[:, None, :]], dim=-1))
        c_crossattn_org = self.cc_projection(torch.cat([clip_emb_1, T_1[:, None, :]], dim=-1))
        cond["c_crossattn"] = [c_crossattn_org]
        latent_img = self.encode_first_stage((xc_1.to(self.device))).mode().detach()
        # c_concat_img = input_mask * latent_img
        c_concat_img = latent_img
        cond["c_concat"] = [c_concat_img]
        
        with torch.enable_grad():
            output_dict = self.Lift3D(batch)
            pred = output_dict["feat"]

        cond["hint"] = [pred]
        out = [pred, z, cond]
        if return_original_cond:
            xrec = self.decode_first_stage(z)
            conrec = self.decode_first_stage(pred)
            depth = output_dict["depth"]
            depth = depth / depth.max()
            depth_img = depth.repeat(1, 3, 1, 1)
            out.extend([x, xrec, conrec, depth_img, xc_1, xc_2, xc_3])

        return out
    
    def configure_optimizers(self):
        lr = self.learning_rate
        print(f'setting learning rate to {lr:.4f} ...')

        params = []
        for name, param in self.control_model.named_parameters():
            if not param.requires_grad:
                print(f"Parameter without grad: {name}")
        for name, param in self.Lift3D.named_parameters():
            if not param.requires_grad:
                print(f"Parameter without grad: {name}")

        params.append({"params": self.control_model.parameters(), "lr": 10 * lr},)
        params.append({"params": self.Lift3D.parameters(), "lr": lr},)

        opt = torch.optim.AdamW(params, lr=lr)

        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt
        
    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, unconditional_guidance_scale=1., unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = dict()
        pred, z, c, x, xrec, conrec, depth_img, xc_1, xc_2, xc_3 = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        # log["con1_recon"] = con1_recon
        log["latent_pred"] = pred
        log["latent_gt"] = z
        log["inputs"] = x
        log["reconstruction"] = xrec
        log["conrec"] = conrec
        log["con_1"] = xc_1
        log["con_2"] = xc_2
        log["con_3"] = xc_3
        log["depth"] = depth_img

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with ema_scope("Sampling"):
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

            if True: # not self.training
                with ema_scope("Sampling"):
                    c_new = c.copy()
                    c_new["zero123"] = torch.zeros(1)
                    samples, z_denoise_row = self.sample_log(cond=c_new,batch_size=N,ddim=use_ddim,
                                                            ddim_steps=ddim_steps,eta=ddim_eta)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
                x_samples = self.decode_first_stage(samples)
                log["samples_zero123"] = x_samples

            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                             ddim_steps=ddim_steps,eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

        if unconditional_guidance_scale > 1.0:
            uc = self.get_unconditional_conditioning(N, unconditional_guidance_label, image_size=x.shape[-1])
            # uc = torch.zeros_like(c)
            uc["hint"] = c["hint"]
            with ema_scope("Sampling with classifier-free guidance"):
                samples_cfg, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                 ddim_steps=ddim_steps, eta=ddim_eta,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=uc,
                                                 )
                x_samples_cfg = self.decode_first_stage(samples_cfg)
                log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
            
            if True:
                with ema_scope("Sampling with classifier-free guidance"):
                    uc["zero123"] = c_new["zero123"]
                    samples_cfg, _ = self.sample_log(cond=c_new, batch_size=N, ddim=use_ddim,
                                                 ddim_steps=ddim_steps, eta=ddim_eta,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=uc,
                                                 )
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
                    x_samples_cfg = self.decode_first_stage(samples_cfg)
                    log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}_zero123"] = x_samples_cfg

        if inpaint:
            # make a simple center square
            b, h, w = z.shape[0], z.shape[2], z.shape[3]
            mask = torch.ones(N, h, w).to(self.device)
            # zeros will be filled in
            mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
            mask = mask[:, None, ...]
            with ema_scope("Plotting Inpaint"):

                samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                            ddim_steps=ddim_steps, x0=z[:N], mask=mask)
            x_samples = self.decode_first_stage(samples.to(self.device))
            log["samples_inpainting"] = x_samples
            log["mask"] = mask

            # outpaint
            mask = 1. - mask
            with ema_scope("Plotting Outpaint"):
                samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                            ddim_steps=ddim_steps, x0=z[:N], mask=mask)
            x_samples = self.decode_first_stage(samples.to(self.device))
            log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
    
    @torch.no_grad()
    def save_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, unconditional_guidance_scale=1., unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = dict()
        pred, z, c, x, xrec, conrec, depth_img, xc_1, xc_2, xc_3 = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        # log["con1_recon"] = con1_recon
        log["latent_pred"] = pred
        log["latent_gt"] = z
        log["inputs"] = x
        log["reconstruction"] = xrec
        log["conrec"] = conrec
        log["con_1"] = xc_1
        log["con_2"] = xc_2
        log["con_3"] = xc_3
        log["depth"] = depth_img

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with ema_scope("Sampling"):
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                             ddim_steps=ddim_steps,eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

        if unconditional_guidance_scale > 1.0:
            uc = self.get_unconditional_conditioning(N, unconditional_guidance_label, image_size=x.shape[-1])
            # uc = torch.zeros_like(c)
            with ema_scope("Sampling with classifier-free guidance"):
                samples_cfg, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                 ddim_steps=ddim_steps, eta=ddim_eta,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=uc,
                                                 )
                x_samples_cfg = self.decode_first_stage(samples_cfg)
                log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
    

    @torch.no_grad()
    def sample(self, batch, cfg_scale, ddim_steps, ddim_eta=1., unconditional_guidance_label=[""]):
        use_ddim = ddim_steps is not None

        # x = self.get_input_DF(batch, k)
        Ts = batch['Ts'].to(memory_format=torch.contiguous_format).float() # (bs, 4, 3)
        T_1 = Ts[..., 0]
        N = Ts.shape[0]
        # print(Ts.shape)
        Ts = rearrange(Ts, 'b c s -> (b s) c')
        batch['Ts'] = Ts
        # x = x.to(self.device)
        # encoder_posterior = self.encode_first_stage(x)
        # z = self.get_first_stage_encoding(encoder_posterior).detach()
        xcs = batch["image_conds"].float()
        B, H, W, C, S = xcs.shape
        xcs = rearrange(xcs, 'b h w c s -> b c h w s')
        xc_1 = xcs[..., 0]
        # xc_2 = xcs[..., 1]
        # xc_3 = xcs[..., 2]
        xcs = rearrange(xcs, 'b c h w s -> (b s) c h w')
        latent_imgs = self.encode_first_stage((xcs.to(self.device))).mode().detach()
        batch["c_concat"] = latent_imgs
    
        cond = {}
        clip_emb_1 = self.get_learned_conditioning(xc_1).detach()
        c_crossattn_org = self.cc_projection(torch.cat([clip_emb_1, T_1[:, None, :]], dim=-1))
        cond["c_crossattn"] = [c_crossattn_org]
        latent_img = self.encode_first_stage((xc_1.to(self.device))).mode().detach()
        c_concat_img = latent_img
        cond["c_concat"] = [c_concat_img]
        
        output_dict = self.Lift3D(batch)
        pred = output_dict["feat"]

        cond["hint"] = [pred]
        # out = [pred, z, cond]

        # N = min(x.shape[0], N)
        # n_row = min(x.shape[0], n_row)
        uc = self.get_unconditional_conditioning(N, unconditional_guidance_label, image_size=xcs.shape[-1])
        # uc = torch.zeros_like(c)
        uc["hint"] = cond["hint"]
        samples_cfg, _ = self.sample_log(cond=cond, batch_size=N, ddim=use_ddim,
                                        ddim_steps=ddim_steps, eta=ddim_eta,
                                        unconditional_guidance_scale=cfg_scale,
                                        unconditional_conditioning=uc,
                                        )
        x_samples_cfg = self.decode_first_stage(samples_cfg)
        
        return x_samples_cfg
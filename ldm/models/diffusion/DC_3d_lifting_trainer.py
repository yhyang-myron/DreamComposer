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
from pytorch_lightning.utilities.distributed import rank_zero_only
from omegaconf import ListConfig

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config, repeat_interleave
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.attention import CrossAttention

from ldm.models.planeencoder import PlaneEncoder, PlaneEncoderGridDepthWeighted, PlaneEncoderGridDepthWeightedRecon, PlaneEncoderDepthWeightedRecon, UNetEncoderWeightedRecon, UNetEncoderWeightedReconGrid
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

class Lifting_3d(pl.LightningModule):
    def __init__(self, unet_config, first_stage_config, scheduler_config):
        super().__init__()
        self.model = UNetEncoderWeightedRecon(unet_config=unet_config)

        self.scale_factor = 0.18215
        self.instantiate_first_stage(first_stage_config)
        self.loss_type = 'l2'
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

    def training_step(self, batch, batch_idx):
        pred, target = self.get_input(batch, 'image_target')
        loss = self.get_loss(pred, target, mean=True)

        prefix = 'train_triplane' if self.training else 'val_triplane'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss_recon': loss.mean()})

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        pred, target = self.get_input(batch, 'image_target')
        loss = self.get_loss(pred, target, mean=True)

        prefix = 'train_triplane' if self.training else 'val_triplane'
        loss_dict = {}
        loss_dict.update({f'{prefix}/loss_recon': loss.mean()})

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        params.append(self.model.parameters())

        opt = torch.optim.AdamW([{"params": self.model.parameters(), "lr": lr}])

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
    
    def get_input(self, batch, k, bs=None, log_images=False):
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
        
        output_dict = self.model(batch)
        pred = output_dict["feat"]
        out = [pred, z]
        if log_images:
            xrec = self.decode_first_stage(z)
            conrec = self.decode_first_stage(pred)
            depth = output_dict["depth"]
            depth = depth / depth.max()
            depth_img = depth.repeat(1, 3, 1, 1)
            out.extend([x, xrec, conrec, depth_img, xc_1, xc_2, xc_3])

        return out

    @torch.no_grad()
    def pseudocolor_image_batch(self, tensor):
        assert tensor.shape[1] == 4
        colors = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0]
        ], device=tensor.device)
        tensor_normalized = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        colored_channels = (tensor_normalized[:, :, :, :, None] * colors[None, :, None, None, :]).sum(dim=1)
        image_batch = torch.clamp(colored_channels, 0, 1)
        
        return image_batch.permute(0, 3, 1, 2)
    
    @torch.no_grad()
    def log_images(self, batch, N=32, n_row=4, split=None,return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, uncond=0.05):
        pred, z, x, xrec, conrec, depth_img, xc_1, xc_2, xc_3 = self.get_input(batch, 'image_target', log_images=True, bs=N)
        log = dict()
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        # print(pred.shape)
        # print(z.shape)
        # exit(0)
        # pred = self.pseudocolor_image_batch(pred)
        # z = self.pseudocolor_image_batch(z)
        # print(pred.shape)
        # print(pred.dtype)
        # exit(0)
        log["latent_pred"] = pred
        log["latent_gt"] = z
        log["inputs"] = x
        log["reconstruction"] = xrec
        log["conrec"] = conrec
        log["con_1"] = xc_1
        log["con_2"] = xc_2
        log["con_3"] = xc_3
        log["depth"] = depth_img

        return log
    
    @torch.no_grad()
    def save_images(self, batch, N=8, n_row=4, split=None,return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, uncond=0.05):
        pred, z, x, xrec, conrec, depth_img, xc_1, xc_2, xc_3 = self.get_input(batch, 'image_target', log_images=True, bs=N)
        log = dict()
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["latent_pred"] = pred
        log["latent_gt"] = z
        log["inputs"] = x
        log["reconstruction"] = xrec
        log["conrec"] = conrec
        log["con_1"] = xc_1
        log["con_2"] = xc_2
        log["con_3"] = xc_3
        log["depth"] = depth_img

        return log
    
    @torch.no_grad()
    def get_input_DF(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)
    
    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z
    
    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):

        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)
    
    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss
import torch
import torch.nn as nn
from einops import rearrange

from ldm.modules.volumetric_rendering.renderer import  PlaneWeightedSamplerAlign
from ldm.modules.volumetric_rendering.ray_sampler import RaySampler
from ldm.util import instantiate_from_config


class Decoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim + 3, hidden_dim), 
                                 nn.Softplus(),
                                 nn.Linear(hidden_dim, output_dim + 1))
    
    def forward(self, sampled_features, ray_directions):
        x = torch.concat((sampled_features, ray_directions), dim=2)

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}


class RenderPlaneWeighted(nn.Module):
    def __init__(self, output_dim=4, neural_rendering_resolution=32):
        super().__init__()
        self.renderer = PlaneWeightedSamplerAlign()
        self.ray_sampler = RaySampler()
        self.decoder = Decoder(input_dim=32, output_dim=output_dim)
        self.neural_rendering_resolution = neural_rendering_resolution
        self.rendering_kwargs = {
            'disparity_space_sampling': False,
            'clamp_mode': 'softplus',
            'depth_resolution': 16,
            'depth_resolution_importance': 16,
            'ray_start': 'auto',
            'ray_end': 'auto',
            'box_warp': 1.6,
            'white_back': True,
            'avg_camera_radius': 1.7,
            'avg_camera_pivot': [0, 0, 0],}
    
    def forward(self, planes, c, Ts, valid_num, con_cameras, lam_org):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, self.neural_rendering_resolution)
        N, M, _ = ray_origins.shape
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs, Ts, valid_num, con_cameras, lam_org) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        return dict(feat=feature_image, depth=depth_image)


class UNetEncoderWeightedRecon(nn.Module):
    def __init__(self, output_dim=4, unet_config=None):
        super().__init__()
        self.encoder = instantiate_from_config(unet_config)
        self.render = RenderPlaneWeighted(output_dim=output_dim)
    
    def forward(self, batch):
        camera = batch["camera"].float()
        imgs = batch["c_concat"].float()
        valid_num = batch["valid_num"].float()
        Ts = batch["Ts"].float()
        con_cameras = batch["con_cameras"].float()
        lam_org = batch["lam_org"].float()
        planes = self.encoder(imgs, Ts)
        output_dict = self.render(planes, camera, Ts, valid_num, con_cameras, lam_org)
        return output_dict
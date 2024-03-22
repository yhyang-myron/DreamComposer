import torch
import torch.nn as nn
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

from ldm.modules.diffusionmodules.openaimodel import UNetModel

class UNetLift3d(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        del self.time_embed
        time_embed_dim = self.model_channels * 4
        self.view_embed = nn.Sequential(
            linear(4, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, x, Ts=None, context=None, y=None,**kwargs):
 
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []

        emb = self.view_embed(Ts)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)
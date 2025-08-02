import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import CompressionModel
from src.entropy_models import EntropyBottleneck, GaussianConditional
from src.ops import ste_round
from compressai.ans import BufferedRansEncoder, RansDecoder
from .utils import update_registered_buffers
from src.layers.elic_ckbd import *
from src.layers.elic_transform import *
from src.layers import CheckboardMaskedConv2d

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)

def model_config():
    config = Config({
        "N": 192,
        "M": 320,
        "slice_num": 10,
        "context_window": 5,
        "slice_ch": [8, 8, 8, 8, 16, 16, 32, 32, 96, 96],
        "quant": "ste",
    })

    return config

class ELIC(CompressionModel):
    def __init__(self, N=192, M=320, num_slices=10, topk=-1, hyper_topk=32, pos='default', hyper_pos='default', patch_size=(256, 256), train_mode=None, **kwargs):
        super().__init__()

        config = model_config()

        N = config.N
        M = config.M
        slice_num = config.slice_num
        slice_ch = config.slice_ch # [8, 8, 8, 8, 16, 16, 32, 32, 96, 96]
        self.quant = config.quant # noise or ste
        self.slice_num = slice_num
        self.slice_ch = slice_ch
        self.g_a = AnalysisTransformEX(N, M, act=nn.ReLU)
        self.g_s = SynthesisTransformEX(N, M, act=nn.ReLU)
        # Hyper Transform
        self.h_a = HyperAnalysisEX(N, M, act=nn.ReLU)
        self.h_s = HyperSynthesisEX(N, M, act=nn.ReLU)
        # Channel Fusion Model
        self.local_context = nn.ModuleList(
            # TODO
        )
        self.channel_context = nn.ModuleList(
            ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2, act=nn.ReLU) if i else None
            for i in range(slice_num)
        )
        # Use channel_ctx and hyper_params
        self.entropy_parameters_anchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i else EntropyParametersEX(in_dim=M * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )
        # Entropy parameters for non-anchors
        # Use spatial_params, channel_ctx and hyper_params
        self.entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            if i else  EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2, act=nn.ReLU)
            for i in range(slice_num)
        )

        # Gussian Conditional
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x, epoch=None):
        # TODO

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    
    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)
    
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x):
        # TODO
        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape):
        # TODO
        return {
            "x_hat": x_hat,
            "cost_time": cost_time
        }

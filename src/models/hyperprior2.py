# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.ops import ste_round
from .base import CompressionModel
from ..entropy_models import EntropyBottleneck
from ..entropy_models import GaussianConditional
from compressai.layers import GDN, MaskedConv2d

from .utils import conv, deconv, update_registered_buffers

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class Hyperprior2(CompressionModel):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """
    
    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(**kwargs)
        
        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv = nn.Sequential(
                    conv(channels, channels, stride=1, kernel_size=3),
                    GDN(channels),
                    nn.LeakyReLU(inplace=True),
                    conv(channels, channels, stride=1, kernel_size=3),
                    GDN(channels)
                )
    
            def forward(self, x):
                return x + self.conv(x) 
            
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=3), 
            ResidualBlock(N),
            conv(N, N, kernel_size=3),
            ResidualBlock(N),
            conv(N, M, kernel_size=3),
        )
        
        self.g_s = nn.Sequential(
            ResidualBlock(N),
            deconv(N, M,kernel_size=3),
            ResidualBlock(N),
            deconv(M, M,kernel_size=3),
            ResidualBlock(N),
            deconv(M, 3,kernel_size=3)
        )
        
        self.h_a = nn.Sequential(
            conv(M,N,stride=1, kernel_size=3),  
            nn.LeakyReLU(inplace=True),
            conv(N, N,stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N,kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N,stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N,kernel_size=3)
        )

        self.h_s = nn.Sequential(
            conv(N, M,kernel_size=3,stride=1),
            nn.LeakyReLU(inplace=True), 
            deconv(M, M*3//2,kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M*3//2, M*3//2,kernel_size=3,stride=1),
            nn.LeakyReLU(inplace=True),
            deconv(M*3//2, M*3//2,kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(M*3//2, M * 2,kernel_size=3,stride=1),
        )
        
        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )
        
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),  
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1), 
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1), 
        )

        self.N = int(N)
        self.M = int(M)
        
        self.gaussian_conditional = GaussianConditional(None)
        self.entropy_bottleneck = EntropyBottleneck(self.N)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(   
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat) 
        x_hat2 = self.g_s(y_hat)
        x_hat,_=self.gaussian_conditional(x,x_hat2)

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
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y,这是因为z = self.h_a(y)进行了2次stride=2的下采样，缩小了4倍
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2 #为了保证输入输出尺寸一致

        y_height = z_hat.size(2)* s
        y_width = z_hat.size(3)* s

        y_hat = F.pad(y, (padding, padding, padding, padding)) #补充边缘

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)
        
        x_hat2 = self.g_s(y_hat)
        x_indexes = self.gaussian_conditional.build_indexes(x_hat2)  #使用x_hat2作为scale
        x_strings = self.gaussian_conditional.compress(x, x_indexes)
        return {
        "strings": [y_strings, z_strings, x_strings],  #x_strings
        "shape": z.size()[-2:],
        "x_hat2_shape": x_hat2.size()[-2:]  
        }

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size] #裁剪处kernel_size大小的块
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)  #移除后两个维度
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding] #获取当前像素
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 3

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )
        
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2)* s
        y_width = z_hat.size(3)* s

        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        ) 
        #（z_hat同样的batch大小，M个通道，高度+左右填充，宽度+上下填充）=>保证尺寸不变
        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        # x_hat = self.g_s(y_hat).clamp_(0, 1)
        
        #解压x_hat
        x_hat2 = self.g_s(y_hat)
        x_indexes = self.gaussian_conditional.build_indexes(x_hat2)
        x_hat = self.gaussian_conditional.decompress(strings[2], x_indexes, x_hat2.dtype)
    
        return {
        "x_hat": x_hat.clamp_(0, 1),  #保持[0,1]之间
        }

    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for h in range(height):
            for w in range(width):
               
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
               
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                ) #从这一步开始解码
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)
                #这些操作都和compressar对应
                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv
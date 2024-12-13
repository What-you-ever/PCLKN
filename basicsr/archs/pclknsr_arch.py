import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import trunc_normal_

import pywt


def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def feature_shuffle(x, index):  
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)  
    index = index.expand(x.shape)  

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x





class GateFFn(nn.Module):
    """
      https://github.com/yuweihao/MambaOut/blob/main/models/mambaout.py
    """

    def __init__(self, dim, kernel_size=5):
        super().__init__()

        self.fc1 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.fc2 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.fc1(x).chunk(2, dim=1)
        x2 = self.dwconv(x2)
        x = F.gelu(x1) * x2
        x = self.fc2(x)
        return x

    def flops(self, h, w, dim):
        flops = 0
        # fc1
        flops += h * w * dim * (dim * 2)
        # dwconv
        flops += h * w * dim * 5 * 5
        # fc2
        flops += h * w * dim * dim
        #  F.gelu(x1) * x2
        flops += h * w * dim
        return flops


class LKA(nn.Module):
    def __init__(self, dim, ksize=5):
        super().__init__()

        if ksize == 9:
            self.conv0 = nn.Conv2d(dim, dim, 3, padding=3 >> 1, groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)
            self.conv1 = nn.Conv2d(dim, dim, 1)
        elif ksize == 13:  # (d-1)*(k-1)+k (3-1)*(5-1)+5=13 p=((k-1)*d)/2=((5-1)*3)/2=6
            self.conv0 = nn.Conv2d(dim, dim, 5, padding=5 >> 1, groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)
            self.conv1 = nn.Conv2d(dim, dim, 1)
        elif ksize == 19:
            self.conv0 = nn.Conv2d(dim, dim, 5, padding=5 >> 1, groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
            self.conv1 = nn.Conv2d(dim, dim, 1)
        elif ksize == 31:
            self.conv0 = nn.Conv2d(dim, dim, 7, padding=7 >> 1, groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, 11, stride=1, padding=15, groups=dim, dilation=3)
            self.conv1 = nn.Conv2d(dim, dim, 1)
        else:
            self.conv0 = nn.Conv2d(dim, dim, 3, 1, 1)
            self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)
            self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return x * attn

    def flops(self, h, w, dim, ksize):
        flops = 0
        # conv0 conv_spatial
        if ksize == 13:
            flops += h * w * dim * 5 * 5
            flops += h * w * dim * 5 * 5
        # conv1
        flops += h * w * dim * dim
        # x*attn
        flops += h * w * dim
        return flops


class LSKA_h(nn.Module):
    def __init__(self, dim, ksize=13):
        super().__init__()

        if ksize == 13:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, 6), groups=dim,
                                            dilation=3)
            self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv_spatial_h(attn)
        attn = self.conv1(attn)
        return u * attn

    def flops(self, h, w, dim):
        flops = 0
        # conv0h
        flops += h * w * dim * 1 * 5
        # spatial_conv
        flops += h * w * dim * 1 * 5
        # conv1
        flops += h * w * dim * dim
        # u*attn
        flops += h * w * dim
        return flops


class ELK(nn.Module):
    def __init__(self, dim=64, k_size=13):
        super().__init__()
        self.k_size = k_size
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(dim=dim, ksize=k_size)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        y = self.proj_1(x)
        y = self.activation(y)
        y = self.spatial_gating_unit(y)
        y = self.proj_2(y)
        x = x + y
        return x

    def flops(self, h, w, dim):
        flops = 0
        # proj_1 proj_2
        flops += h * w * dim * dim * 2
        # LKA
        flops += self.spatial_gating_unit.flops(h, w, dim, self.k_size)
        return flops


class MPC(nn.Module):


    def __init__(self, dim, category_size=[128, 128, 64, 64, 32, 32], i_block=2, i_group=0):
        super().__init__()
        self.category_size = category_size[i_block]
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKA_h(dim, ksize=13)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x, sa): 
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()
        sim = sa.permute(0, 2, 3, 1).reshape(b, h * w).contiguous()
        n = h * w
        gs = min(n, self.category_size)  
        ng = (n + gs - 1) // gs  
        x_sort_values, x_sort_indices = torch.sort(sim, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)
        shuffled_x = feature_shuffle(x, x_sort_indices)  #
        pad_n = ng * gs - n
        paded_x = torch.cat((shuffled_x, torch.flip(shuffled_x[:, n - pad_n:n, :], dims=[1])),
                            dim=1)  
        y = paded_x.reshape(b, -1, gs, c).contiguous()  
        y = y.permute(0, 3, 2, 1).contiguous()  
        y = self.proj_1(y)
        y = self.activation(y)
        y = self.spatial_gating_unit(y)
        y = self.proj_2(y)
        y = y.permute(0, 3, 2, 1).reshape(b, n, c).contiguous()
        y = feature_shuffle(y, x_sort_indices_reverse)
        y = x + y
        y = y.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        return y

    def flops(self, h, w, dim):
        flops = 0
        # proj_1 proj_2
        flops += h * w * dim * dim * 2
        # lska_h
        flops += self.spatial_gating_unit.flops(h, w, dim)

        return flops


class SCAttention(nn.Module):
    def __init__(self, dim=60, k=3):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim + k, dim // 3, 1),
            LayerNorm(dim // 3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim // 3, dim, 1),
            nn.Sigmoid()
        )

        self.sa = nn.Sequential(
            nn.Conv2d(dim // 3, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, global_info):
        x = torch.cat([x, global_info], dim=1)
        x = self.conv1(x)
        sa = self.sa(x)
        ca = self.ca(x)
        return sa, ca

    def flops(self, h, w, dim):
        flops = 0
        # conv1
        flops += h * w * (dim + 3) * (dim // 3)
        # ca
        flops += h * w + (dim // 3) * dim
        # sa
        flops += h * w * (dim // 3) * 3 * 3
        return flops


class PCLKB(nn.Module):
    def __init__(self, n_feats=60, k_size=13, i_group=0, i_block=0):
        super().__init__()

        self.head = nn.Conv2d(n_feats, n_feats, 1)
        self.lka = ELK(dim=n_feats, k_size=k_size)
        self.sca = SCAttention(dim=n_feats)
        self.diva = MPC(dim=n_feats, i_group=i_group, i_block=i_block)
        self.alpha = nn.Parameter(torch.ones(2), requires_grad=True)
        self.norm1 = LayerNorm(n_feats)
        self.norm2 = LayerNorm(n_feats)
        self.ffn = GateFFn(n_feats)

    def forward(self, x, global_info):
        res = x.clone()
        y = self.head(x)
        sa, ca = self.sca(y, global_info)
        alpha = torch.clamp(self.alpha, -0.3, 1.2)  
        z = y * sa + y
        lka = self.lka(z) * ca
        diva = self.diva(y, sa) * ca  
        x = y + alpha[0] * lka + alpha[1] * diva
        x = self.norm1(x)
        x = self.ffn(x) + res
        x = self.norm2(x)
        return x

    def flops(self, h, w, dim):
        flops = 0
        # head
        flops += h * w * dim * dim
        # elk
        flops += self.lka.flops(h, w, dim)
        # sca
        flops += self.sca.flops(h, w, dim)
        # mpc
        flops += self.diva.flops(h, w, dim)
        # ffn
        flops += self.ffn.flops(h, w, dim)
        # norm1 norm2
        flops += h * w * dim * 2
        # forward
        flops += h * w * dim * 3
        return flops


class PCLK(nn.Module):
    def __init__(self, n_feats=60, i_group=0, num_blocks=[4, 4, 6, 6], k_size=13):
        super().__init__()

        self.body = nn.ModuleList([PCLKB(n_feats=n_feats, k_size=k_size, i_group=i_group, i_block=i) for i in
                                   range(num_blocks[i_group])])
        self.tail = nn.Conv2d(n_feats, n_feats, 1)

    def forward(self, x, global_info):
        res = x.clone()
        for block in self.body:
            x = block(x, global_info)
        x = self.tail(x)
        return x + res

    def flops(self, h, w, dim):
        flops = 0
        # body
        for layer in self.body:
            flops += layer.flops(h, w, dim)
        # tail
        flops += h * w * dim * dim

        return flops


@ARCH_REGISTRY.register()
class PCLKNSR(nn.Module):
    def __init__(self, c_colors=3, num_features=42, scale=4, img_range=1, num_groups=6, num_blocks=[6, 6, 6, 6, 6, 6],
                 k_size=13, batch_size=32, img_size=64):
        super().__init__()
        self.scale = scale
        self.img_range = img_range
        self.num_features = num_features
        self.img_size = img_size

        # 0: data pre-process (mean-std)
        self.mean = torch.tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1)

        # 1: shallow feature extraction
        self.head = nn.Conv2d(c_colors, num_features, 3, 1, 1)
        self.global_info = nn.Sequential(nn.Conv2d(num_features, 9, 1),
                                         nn.LeakyReLU(0.1, inplace=True),
                                         nn.Conv2d(9, 3, 3, 1, 1),
                                         nn.LeakyReLU(0.1, inplace=True))

        self.body = nn.ModuleList([PCLK(
            n_feats=num_features,
            i_group=i,
            k_size=k_size,
            num_blocks=num_blocks
        ) for i in range(num_groups)])
        self.body_tail = nn.Conv2d(num_features, num_features, 3, 1, 1)

        # 3: image reconstruction
        self.tail = nn.Sequential(
            nn.Conv2d(num_features, c_colors * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )


    def forward(self, x):
        h_ori, w_ori = x.size()[-2], x.size()[-1]  #
        x = self.img_padding(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.head(x)
        global_info = self.global_info(x)
        res = x.clone()

        for group in self.body:
            x = group(x, global_info)
        x = self.body_tail(x)
        x = self.tail(x + res)
        x = x / self.img_range + self.mean
        # unpadding
        x = x[:, :, :h_ori * self.scale, :w_ori * self.scale]

        return x

    def img_padding(self, x):
        h, w = x.size()[-2:]
        if h != self.img_size or w != self.img_size:
            mod = self.img_size
            h_pad = ((h + mod - 1) // mod) * mod - h
            w_pad = ((w + mod - 1) // mod) * mod - w
            h, w = h + h_pad, w + w_pad

            x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
            x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]


        return x

    def flops(self, h, w):
        c = 3
        flops = 0
        # head
        flops += h * w * c * self.num_features * 3 * 3
        # global_info
        flops += h * w * self.num_features * 9 * 3 * 3 + h * w * 9 * 3 * 3 * 3
        # body
        for layer in self.body:
            flops += layer.flops(h, w, self.num_features)
        # body_tail
        flops += h * w * self.num_features * self.num_features * 3 * 3
        # tail
        flops += h * w * self.num_features * (self.scale ** 2) * 3 * 3 * 3
        return flops

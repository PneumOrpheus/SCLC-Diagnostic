# --------------------------------------------------------
# Swin Transformer V2 - 3D Extension
# Extends 2D Swin Transformer V2 to operate on volumetric (3D) data
# with 3D window partitioning, shifted windows, and 3D relative position bias.
#
# Based on: Swin Transformer V2 (Copyright (c) 2022 Microsoft, MIT License)
# 3D adaptation for CT volume processing in SCLC classification.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, trunc_normal_
import numpy as np
from typing import Tuple


def to_3tuple(x):
    """Convert scalar or sequence to 3-tuple."""
    if isinstance(x, (list, tuple)):
        if len(x) == 3:
            return tuple(x)
        elif len(x) == 2:
            return (x[0], x[1], x[1])
        elif len(x) == 1:
            return (x[0], x[0], x[0])
    return (x, x, x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition_3d(x, window_size):
    """Partition a 3D volume into non-overlapping windows.

    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): (Wd, Wh, Ww)

    Returns:
        windows: (num_windows*B, Wd, Wh, Ww, C)
    """
    B, D, H, W, C = x.shape
    Wd, Wh, Ww = window_size
    x = x.view(B, D // Wd, Wd, H // Wh, Wh, W // Ww, Ww, C)
    # (B, nD, Wd, nH, Wh, nW, Ww, C) -> (B, nD, nH, nW, Wd, Wh, Ww, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, Wd, Wh, Ww, C)
    return windows


def window_reverse_3d(windows, window_size, D, H, W):
    """Reverse 3D window partitioning.

    Args:
        windows: (num_windows*B, Wd, Wh, Ww, C)
        window_size (tuple[int]): (Wd, Wh, Ww)
        D, H, W: original volume dimensions

    Returns:
        x: (B, D, H, W, C)
    """
    Wd, Wh, Ww = window_size
    nD, nH, nW = D // Wd, H // Wh, W // Ww
    B = int(windows.shape[0] / (nD * nH * nW))
    x = windows.view(B, nD, nH, nW, Wd, Wh, Ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class WindowAttention3D(nn.Module):
    """3D Window based multi-head self attention with continuous relative position bias (V2).

    Computes self-attention within non-overlapping 3D windows using cosine attention
    and a learned continuous relative position bias via an MLP.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): 3D window size (Wd, Wh, Ww).
        num_heads (int): Number of attention heads.
        qkv_bias (bool): If True, add learnable bias to Q and V.
        attn_drop (float): Dropout ratio of attention weight.
        proj_drop (float): Dropout ratio of output.
        pretrained_window_size (tuple[int]): Window size used in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=(0, 0, 0)):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wd, Wh, Ww)
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # MLP to generate continuous relative position bias (3D: input is 3 coords)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        # Build 3D relative coords table
        relative_coords_d = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_h = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[2] - 1), self.window_size[2], dtype=torch.float32)
        # Shape: (2*Wd-1, 2*Wh-1, 2*Ww-1, 3)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_d, relative_coords_h, relative_coords_w], indexing='ij')
        ).permute(1, 2, 3, 0).contiguous().unsqueeze(0)

        # Normalize coordinates
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, :, 1] /= (pretrained_window_size[1] - 1)
            relative_coords_table[:, :, :, :, 2] /= (pretrained_window_size[2] - 1)
        else:
            relative_coords_table[:, :, :, :, 0] /= (self.window_size[0] - 1) if self.window_size[0] > 1 else 1
            relative_coords_table[:, :, :, :, 1] /= (self.window_size[1] - 1) if self.window_size[1] > 1 else 1
            relative_coords_table[:, :, :, :, 2] /= (self.window_size[2] - 1) if self.window_size[2] > 1 else 1

        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # 3D pair-wise relative position index
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w], indexing='ij'))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 3

        # Shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        # Compute flattened index
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, N, C) where N = Wd*Wh*Ww
            mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=np.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        # 3D relative position bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1)  # N, N, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, N, N
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return (f'dim={self.dim}, window_size={self.window_size}, '
                f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}')


class SwinTransformerBlock3D(nn.Module):
    """3D Swin Transformer Block with shifted 3D window attention.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): (D, H, W) input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): 3D window size (Wd, Wh, Ww).
        shift_size (tuple[int]): 3D shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add learnable bias to Q, V.
        drop (float): Dropout rate.
        attn_drop (float): Attention dropout rate.
        drop_path (float): Stochastic depth rate.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
        pretrained_window_size (tuple[int]): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=(2, 7, 7),
                 shift_size=(0, 0, 0), mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 pretrained_window_size=(0, 0, 0)):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (D, H, W)
        self.num_heads = num_heads
        self.window_size = list(window_size)
        self.shift_size = list(shift_size)
        self.mlp_ratio = mlp_ratio

        # Adjust window size if input is smaller
        for i in range(3):
            if self.input_resolution[i] <= self.window_size[i]:
                self.shift_size[i] = 0
                self.window_size[i] = self.input_resolution[i]

        self.window_size = tuple(self.window_size)
        self.shift_size = tuple(self.shift_size)

        assert all(0 <= s < w for s, w in zip(self.shift_size, self.window_size)), \
            "shift_size must be in [0, window_size)"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_3tuple(pretrained_window_size) if not isinstance(pretrained_window_size, tuple) else pretrained_window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Compute 3D attention mask for shifted windows
        if any(s > 0 for s in self.shift_size):
            D, H, W = self.input_resolution
            Wd, Wh, Ww = self.window_size
            Sd, Sh, Sw = self.shift_size

            img_mask = torch.zeros((1, D, H, W, 1))
            d_slices = (slice(0, -Wd), slice(-Wd, -Sd), slice(-Sd, None))
            h_slices = (slice(0, -Wh), slice(-Wh, -Sh), slice(-Sh, None))
            w_slices = (slice(0, -Ww), slice(-Ww, -Sw), slice(-Sw, None))

            cnt = 0
            for d in d_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, d, h, w, :] = cnt
                        cnt += 1

            mask_windows = window_partition_3d(img_mask, self.window_size)  # nW, Wd, Wh, Ww, 1
            mask_windows = mask_windows.view(-1, Wd * Wh * Ww)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        D, H, W = self.input_resolution
        Wd, Wh, Ww = self.window_size
        B, L, C = x.shape
        assert L == D * H * W, f"input feature has wrong size: {L} vs {D}*{H}*{W}={D*H*W}"

        shortcut = x
        x = x.view(B, D, H, W, C)

        # Cyclic shift in 3D
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x,
                                   shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                   dims=(1, 2, 3))
        else:
            shifted_x = x

        # Partition into 3D windows
        x_windows = window_partition_3d(shifted_x, self.window_size)  # nW*B, Wd, Wh, Ww, C
        x_windows = x_windows.view(-1, Wd * Wh * Ww, C)  # nW*B, N, C

        # 3D W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, N, C

        # Merge windows
        attn_windows = attn_windows.view(-1, Wd, Wh, Ww, C)
        shifted_x = window_reverse_3d(attn_windows, self.window_size, D, H, W)  # B, D, H, W, C

        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(shifted_x,
                           shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                           dims=(1, 2, 3))
        else:
            x = shifted_x

        x = x.view(B, D * H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, input_resolution={self.input_resolution}, "
                f"num_heads={self.num_heads}, window_size={self.window_size}, "
                f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}")


class PatchMerging3D(nn.Module):
    """3D Patch Merging Layer.

    Downsamples spatial dimensions (H, W) by 2x while preserving depth (D).
    This keeps the depth dimension intact for continued 3D attention while
    reducing spatial resolution like the 2D version.

    Args:
        input_resolution (tuple[int]): (D, H, W) input resolution.
        dim (int): Number of input channels.
        norm_layer: Normalization layer.
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # Merge 2x2 spatial patches (same as 2D), keeping depth
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        D, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, D, H, W, C)

        # Merge 2x2 spatial patches (keep D intact)
        x0 = x[:, :, 0::2, 0::2, :]  # B, D, H/2, W/2, C
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # B, D, H/2, W/2, 4*C
        x = x.view(B, -1, 4 * C)  # B, D*H/2*W/2, 4*C

        x = self.reduction(x)
        x = self.norm(x)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer3D(nn.Module):
    """A basic 3D Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): (D, H, W) input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): 3D window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add learnable bias to Q, V.
        drop (float): Dropout rate.
        attn_drop (float): Attention dropout rate.
        drop_path (float | list[float]): Stochastic depth rate.
        norm_layer: Normalization layer.
        downsample: Downsample layer at the end of the layer.
        use_checkpoint (bool): Whether to use gradient checkpointing.
        pretrained_window_size (tuple[int]): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=(0, 0, 0)):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (D, H, W)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        window_size = to_3tuple(window_size) if not isinstance(window_size, tuple) else window_size

        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (
                    window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=to_3tuple(pretrained_window_size) if not isinstance(pretrained_window_size, tuple) else pretrained_window_size)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed3D(nn.Module):
    """3D Volume to Patch Embedding.

    Uses Conv3d to project volumetric input into patch tokens.

    Args:
        img_size (int): Spatial image size (H, W). Default: 224.
        depth_size (int): Depth (number of slices). Default: 16.
        patch_size (int): Spatial patch size. Default: 4.
        depth_patch_size (int): Depth patch size. Default: 2.
        in_chans (int): Number of input channels. Default: 1 (CT grayscale).
        embed_dim (int): Embedding dimension. Default: 96.
        norm_layer: Normalization layer.
    """

    def __init__(self, img_size=224, depth_size=16, patch_size=4, depth_patch_size=2,
                 in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        self.depth_size = depth_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        self.depth_patch_size = depth_patch_size

        self.patches_resolution = (
            depth_size // depth_patch_size,
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1]
        )
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1] * self.patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(depth_patch_size, patch_size, patch_size),
            stride=(depth_patch_size, patch_size, patch_size)
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, Pd*Ph*Pw, embed_dim)
        """
        B, C, D, H, W = x.shape
        assert D == self.depth_size and H == self.img_size[0] and W == self.img_size[1], \
            f"Input size ({D}*{H}*{W}) doesn't match model ({self.depth_size}*{self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)  # B, embed_dim, Pd, Ph, Pw
        x = x.flatten(2).transpose(1, 2)  # B, Pd*Ph*Pw, embed_dim
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformerV2_3D(nn.Module):
    """3D Swin Transformer V2 for volumetric CT data.

    Processes 3D volumes with hierarchical 3D window attention.
    Outputs 2D feature maps (depth collapsed via pooling) for compatibility
    with the existing FPN and detection pipeline.

    Args:
        img_size (int): Spatial image size. Default: 224.
        depth_size (int): Number of depth slices. Default: 16.
        patch_size (int): Spatial patch size. Default: 4.
        depth_patch_size (int): Depth patch size. Default: 2.
        in_chans (int): Input channels. Default: 1 (grayscale CT).
        num_classes (int): Number of output classes. Default: 1000.
        embed_dim (int): Patch embedding dimension. Default: 96.
        depths (list[int]): Depth of each stage. Default: [2, 2, 6, 2].
        num_heads (list[int]): Number of attention heads per stage.
        window_size (int | tuple): 3D window size. Default: (2, 7, 7).
        mlp_ratio (float): MLP ratio. Default: 4.
        qkv_bias (bool): Add bias to Q, V. Default: True.
        drop_rate (float): Dropout rate. Default: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.1.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        ape (bool): Absolute position embedding. Default: False.
        patch_norm (bool): Normalize after patch embedding. Default: True.
        use_checkpoint (bool): Gradient checkpointing. Default: False.
        pretrained_window_sizes (list[int]): Pre-trained window sizes.
    """

    def __init__(self, img_size=224, depth_size=16, patch_size=4, depth_patch_size=2,
                 in_chans=1, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=(2, 7, 7), mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=(0, 0, 0, 0), **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # Convert window_size to 3-tuple
        if isinstance(window_size, int):
            window_size = (min(window_size, depth_size // depth_patch_size), window_size, window_size)

        self.patch_embed = PatchEmbed3D(
            img_size=img_size, depth_size=depth_size,
            patch_size=patch_size, depth_patch_size=depth_patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution  # (Pd, Ph, Pw)
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            D_i = patches_resolution[0]  # Depth stays constant (no depth downsampling)
            H_i = patches_resolution[1] // (2 ** i_layer)
            W_i = patches_resolution[2] // (2 ** i_layer)

            # Adjust window size for this resolution
            layer_window_size = (
                min(window_size[0], D_i),
                min(window_size[1], H_i),
                min(window_size[2], W_i)
            )

            # Ensure input resolution is divisible by window size
            for dim_idx in range(3):
                res = [D_i, H_i, W_i][dim_idx]
                ws = layer_window_size[dim_idx]
                if res % ws != 0:
                    # Find largest divisor <= ws
                    for candidate in range(ws, 0, -1):
                        if res % candidate == 0:
                            layer_window_size = list(layer_window_size)
                            layer_window_size[dim_idx] = candidate
                            layer_window_size = tuple(layer_window_size)
                            break

            pws = pretrained_window_sizes[i_layer] if isinstance(pretrained_window_sizes[i_layer], tuple) else to_3tuple(pretrained_window_sizes[i_layer])

            layer = BasicLayer3D(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(D_i, H_i, W_i),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=layer_window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging3D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                pretrained_window_size=pws)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B, L, C
        x = self.avgpool(x.transpose(1, 2))  # B, C, 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

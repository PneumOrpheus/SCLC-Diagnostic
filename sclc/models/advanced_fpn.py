import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _conv(spatial_dims: int):
    if spatial_dims == 2:
        return nn.Conv2d
    if spatial_dims == 3:
        return nn.Conv3d
    raise ValueError(f"Unsupported spatial_dims={spatial_dims}")


def _lazy_conv(spatial_dims: int):
    if spatial_dims == 2:
        return nn.LazyConv2d
    if spatial_dims == 3:
        return nn.LazyConv3d
    raise ValueError(f"Unsupported spatial_dims={spatial_dims}")


def _adaptive_pool(spatial_dims: int, mode: str = "avg"):
    if spatial_dims == 2:
        return nn.AdaptiveAvgPool2d if mode == "avg" else nn.AdaptiveMaxPool2d
    if spatial_dims == 3:
        return nn.AdaptiveAvgPool3d if mode == "avg" else nn.AdaptiveMaxPool3d
    raise ValueError(f"Unsupported spatial_dims={spatial_dims}")


def _interpolate(x: torch.Tensor, size: Sequence[int], spatial_dims: int):
    mode = "nearest" if spatial_dims == 2 else "nearest"
    return F.interpolate(x, size=size, mode=mode)


# -----------------------------------------------------------------------------
# Attention + scale modules
# -----------------------------------------------------------------------------


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_dims: int = 2):
        super().__init__()
        reduced = max(1, channels // reduction)
        Conv = _conv(spatial_dims)
        self.avg_pool = _adaptive_pool(spatial_dims, mode="avg")(1)
        self.max_pool = _adaptive_pool(spatial_dims, mode="max")(1)
        self.mlp = nn.Sequential(
            Conv(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            Conv(reduced, channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(self.avg_pool(x))
        mx = self.mlp(self.max_pool(x))
        gate = torch.sigmoid(avg + mx)
        return x * gate


class ScalePerceptionModule(nn.Module):
    def __init__(self, channels: int, spatial_dims: int = 2, gate_kernel: int = 3):
        super().__init__()
        Conv = _conv(spatial_dims)
        self.conv_small = Conv(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv_large = Conv(channels, channels, kernel_size=3, padding=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.pool = _adaptive_pool(spatial_dims, mode="avg")(1)
        # Produce 2 * C channel gates; softmax over the branch dimension.
        self.gate = Conv(channels, channels * 2, kernel_size=1, bias=True)
        self.gate_kernel = gate_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.conv_small(x))
        x2 = self.act(self.conv_large(x1))
        pooled = self.pool(x1 + x2)
        gates = self.gate(pooled)
        # Reshape to (B, 2, C, 1, 1[, 1]) and softmax over the 2 branches.
        B, C2 = gates.shape[:2]
        C = C2 // 2
        view_shape = (B, 2, C) + (1,) * (gates.ndim - 2)
        gates = gates.view(view_shape)
        weights = torch.softmax(gates, dim=1)
        w1, w2 = weights[:, 0], weights[:, 1]
        out = x1 * w1 + x2 * w2
        return out + x


# -----------------------------------------------------------------------------
# Transformer-style pyramid refinement
# -----------------------------------------------------------------------------


def _positional_encoding_2d(h: int, w: int, channels: int, device: torch.device) -> torch.Tensor:
    if channels % 4 != 0:
        channels = channels - (channels % 4)
    y = torch.linspace(0, 1, h, device=device)
    x = torch.linspace(0, 1, w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    pos = torch.stack([yy, xx], dim=-1).reshape(-1, 2)  # (H*W, 2)
    div = torch.exp(
        torch.arange(0, channels // 2, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / (channels // 2))
    )
    pe = torch.zeros(pos.shape[0], channels, device=device)
    pe[:, 0::4] = torch.sin(pos[:, 0:1] * div)
    pe[:, 1::4] = torch.cos(pos[:, 0:1] * div)
    pe[:, 2::4] = torch.sin(pos[:, 1:2] * div)
    pe[:, 3::4] = torch.cos(pos[:, 1:2] * div)
    return pe


def _positional_encoding_3d(d: int, h: int, w: int, channels: int, device: torch.device) -> torch.Tensor:
    if channels % 6 != 0:
        channels = channels - (channels % 6)
    z = torch.linspace(0, 1, d, device=device)
    y = torch.linspace(0, 1, h, device=device)
    x = torch.linspace(0, 1, w, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    pos = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)  # (D*H*W, 3)
    div = torch.exp(
        torch.arange(0, channels // 3, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / (channels // 3))
    )
    pe = torch.zeros(pos.shape[0], channels, device=device)
    pe[:, 0::6] = torch.sin(pos[:, 0:1] * div)
    pe[:, 1::6] = torch.cos(pos[:, 0:1] * div)
    pe[:, 2::6] = torch.sin(pos[:, 1:2] * div)
    pe[:, 3::6] = torch.cos(pos[:, 1:2] * div)
    pe[:, 4::6] = torch.sin(pos[:, 2:3] * div)
    pe[:, 5::6] = torch.cos(pos[:, 2:3] * div)
    return pe


class TFPNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        spatial_dims: int = 2,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
        pool_size: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.pool_size = pool_size
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
            norm1 = nn.LayerNorm(channels)
            norm2 = nn.LayerNorm(channels)
            mlp = nn.Sequential(
                nn.Linear(channels, channels * 4),
                nn.GELU(),
                nn.Linear(channels * 4, channels),
            )
            self.layers.append(nn.ModuleList([attn, norm1, norm2, mlp]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_size is not None:
            pooled = _interpolate(x, size=self.pool_size, spatial_dims=self.spatial_dims)
        else:
            pooled = x

        B, C = pooled.shape[:2]
        spatial = pooled.shape[2:]
        tokens = pooled.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)

        if self.spatial_dims == 2:
            pos = _positional_encoding_2d(spatial[0], spatial[1], C, x.device)
        else:
            pos = _positional_encoding_3d(spatial[0], spatial[1], spatial[2], C, x.device)
        if pos.shape[1] != C:
            pos = F.pad(pos, (0, C - pos.shape[1]))
        tokens = tokens + pos.unsqueeze(0)

        for attn, norm1, norm2, mlp in self.layers:
            attn_out, _ = attn(tokens, tokens, tokens)
            tokens = norm1(tokens + attn_out)
            mlp_out = mlp(tokens)
            tokens = norm2(tokens + mlp_out)

        out = tokens.permute(0, 2, 1).contiguous().view(B, C, *spatial)
        if self.pool_size is not None and out.shape[2:] != x.shape[2:]:
            out = _interpolate(out, size=x.shape[2:], spatial_dims=self.spatial_dims)
        return x + out


# -----------------------------------------------------------------------------
# Feature fusion
# -----------------------------------------------------------------------------


class MBFFM(nn.Module):
    def __init__(self, channels: int, num_levels: int, spatial_dims: int = 2):
        super().__init__()
        Conv = _conv(spatial_dims)
        self.spatial_dims = spatial_dims
        self.downsamples = nn.ModuleList([
            Conv(channels, channels, kernel_size=3, stride=2, padding=1)
            for _ in range(num_levels - 1)
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        # Top-down
        td = [None] * len(feats)
        td[-1] = feats[-1]
        for i in range(len(feats) - 2, -1, -1):
            up = _interpolate(td[i + 1], size=feats[i].shape[2:], spatial_dims=self.spatial_dims)
            td[i] = feats[i] + up

        # Bottom-up
        out = [None] * len(feats)
        out[0] = td[0]
        for i in range(1, len(feats)):
            down = self.downsamples[i - 1](out[i - 1])
            out[i] = td[i] + down
        return out


class AdvancedFPNNeck(nn.Module):
    def __init__(
        self,
        num_levels: int,
        out_channels: int = 256,
        spatial_dims: int = 2,
        use_tfpn: bool = True,
        tfpn_heads: int = 4,
        tfpn_layers: int = 1,
        tfpn_pool: Optional[Sequence[int]] = None,
        tfpn_levels: int = 1,
        cam_reduction: int = 16,
    ):
        super().__init__()
        LazyConv = _lazy_conv(spatial_dims)
        self.spatial_dims = spatial_dims
        self.out_channels = out_channels
        self.use_tfpn = use_tfpn
        self.tfpn_levels = max(0, int(tfpn_levels))

        self.lateral_convs = nn.ModuleList([
            LazyConv(out_channels, kernel_size=1) for _ in range(num_levels)
        ])
        self.spm = nn.ModuleList([
            ScalePerceptionModule(out_channels, spatial_dims=spatial_dims) for _ in range(num_levels)
        ])
        self.cam = nn.ModuleList([
            ChannelAttention(out_channels, reduction=cam_reduction, spatial_dims=spatial_dims)
            for _ in range(num_levels)
        ])
        if use_tfpn:
            self.tfpn = nn.ModuleList([
                TFPNBlock(
                    out_channels,
                    spatial_dims=spatial_dims,
                    num_heads=tfpn_heads,
                    num_layers=tfpn_layers,
                    pool_size=tfpn_pool,
                )
                for _ in range(num_levels)
            ])
        else:
            self.tfpn = None

        self.mbffm = MBFFM(out_channels, num_levels=num_levels, spatial_dims=spatial_dims)

    def forward(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if len(feats) != len(self.lateral_convs):
            raise ValueError(f"Expected {len(self.lateral_convs)} feature levels, got {len(feats)}")

        out: List[torch.Tensor] = []
        for idx, x in enumerate(feats):
            x = self.lateral_convs[idx](x)
            x = self.spm[idx](x)
            x = self.cam[idx](x)
            out.append(x)

        if self.use_tfpn and self.tfpn is not None and self.tfpn_levels > 0:
            start = max(0, len(out) - self.tfpn_levels)
            for i in range(start, len(out)):
                out[i] = self.tfpn[i](out[i])

        fused = self.mbffm(out)
        return fused[0], fused


class MultiTaskHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        spatial_dims: int = 2,
        use_seg: bool = False,
        use_det: bool = False,
    ):
        super().__init__()
        self.use_seg = use_seg
        self.use_det = use_det
        Pool = _adaptive_pool(spatial_dims, mode="avg")
        Conv = _conv(spatial_dims)
        self.pool = Pool(1)
        self.cls_head = nn.Linear(in_channels, num_classes)
        self.seg_head = Conv(in_channels, 1, kernel_size=1) if use_seg else None
        self.box_head = nn.Linear(in_channels, 4 if spatial_dims == 2 else 6) if use_det else None

    def forward(self, feat_map: torch.Tensor, input_shape: Sequence[int]):
        pooled = self.pool(feat_map).flatten(1)
        cls_logits = self.cls_head(pooled)
        seg_logits = None
        box_pred = None
        if self.use_seg and self.seg_head is not None:
            seg_logits = self.seg_head(feat_map)
            if seg_logits.shape[2:] != tuple(input_shape[2:]):
                seg_logits = _interpolate(seg_logits, size=input_shape[2:], spatial_dims=feat_map.ndim - 2)
        if self.use_det and self.box_head is not None:
            box_pred = torch.sigmoid(self.box_head(pooled))
        return cls_logits, seg_logits, box_pred

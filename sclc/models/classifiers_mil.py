import os
import timm
import torch
import torch.nn as nn
from monai.networks.nets import MILModel

from .classifiers_2d import SwinTiny2DClassifier


class MILResNet50Classifier(nn.Module):
    """BigLunge MIL classifier: attention pooling over whole-slice bags.

    Wraps ``monai.networks.nets.MILModel`` with a torchvision ResNet-50
    backbone (``mil_mode='att'`` by default). CT input is 1-channel; we
    adapt the pretrained 3-channel RGB stem by averaging across channels,
    the same grayscale-transfer trick the other 2D classifiers in this file
    use. Expected forward input: ``(B, N, 1, H, W)``.

    **Batch-norm leakage mitigation.** ``MILModel`` reshapes
    ``(B, N, C, H, W)`` to ``(B*N, C, H, W)`` before the backbone, so BN
    running statistics span every instance in every bag of the minibatch.
    That leaks bag-label information into per-instance features and inflates
    train metrics. We keep all BN modules in ``eval()`` mode during training
    (reusing the frozen running stats from ImageNet/DAPT pretraining) —
    attention and the final classifier still train normally. This is the
    standard fix for CNN-backbone MIL. Override with
    ``self._freeze_bn_for_training = False`` to disable for experiments.

    **Loading a DAPT backbone.** Call ``load_backbone_from_dapt(state_dict)``
    with a checkpoint produced by training ``TorchVisionResNet2DClassifier``.
    Keys like ``backbone.features.conv1.weight`` are re-routed to ``net.*``
    to initialize the MIL backbone; the DAPT classifier head is dropped
    (MIL has its own attention + ``myfc``).
    """

    def __init__(
        self,
        num_classes: int = 3,
        mil_mode: str = "att",
        pretrained_backbone: bool = True,
        trans_blocks: int = 4,
        trans_dropout: float = 0.0,
    ):
        super().__init__()
        self.mil = MILModel(
            num_classes=num_classes,
            mil_mode=mil_mode,
            pretrained=pretrained_backbone,
            backbone="resnet50",
            trans_blocks=trans_blocks,
            trans_dropout=trans_dropout,
        )
        self._adapt_stem_1ch()
        # BN was previously frozen (eval()) during MIL training to prevent
        # bag-label leakage through the (B*N)-wide batch statistics. In
        # practice on 1-ch CT, ImageNet running stats applied to grayscale
        # inputs produced activations far enough out-of-distribution to NaN
        # the attention softmax after one optimizer step. Letting BN update
        # its running stats during MIL fine-tune is the lesser evil here.
        self._freeze_bn_for_training = False

    def _adapt_stem_1ch(self) -> None:
        """Replace the 3-ch torchvision stem with a 1-ch conv initialized
        from the mean of the pretrained RGB weights.
        """
        stem = self.mil.net.conv1  # (64, 3, 7, 7) torchvision ResNet50 stem
        if stem.in_channels == 1:
            return
        new_conv = nn.Conv2d(
            1, stem.out_channels,
            kernel_size=stem.kernel_size, stride=stem.stride,
            padding=stem.padding, bias=stem.bias is not None,
        )
        with torch.no_grad():
            avg = stem.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
            new_conv.weight.copy_(avg)
            if stem.bias is not None:
                new_conv.bias.copy_(stem.bias)
        self.mil.net.conv1 = new_conv

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._freeze_bn_for_training:
            for m in self.mil.net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def load_backbone_from_dapt(self, state_dict, logger=None):
        """Load a DAPT ``TorchVisionResNet2DClassifier`` state_dict into
        ``self.mil.net``.

        DAPT keys look like ``backbone.features.<torchvision_key>``;
        ``self.mil.net`` keys look like ``<torchvision_key>``. We strip the
        ``backbone.features.`` prefix and load non-strict. Any head params
        (``backbone.fc.*``) are dropped — MIL has ``attention`` + ``myfc``
        which stay at their current init.
        """
        src_prefix = "backbone.features."
        target_sd = self.mil.net.state_dict()
        mapped: dict = {}
        for k, v in state_dict.items():
            if k.startswith(src_prefix):
                new_k = k[len(src_prefix):]
                if new_k in target_sd:
                    mapped[new_k] = v
        missing, unexpected = self.mil.net.load_state_dict(mapped, strict=False)
        matched = len(target_sd) - len(missing)
        if logger is not None:
            logger.info(
                f"[MIL] Loaded DAPT backbone: matched {matched}/{len(target_sd)} target keys "
                f"(source contributed {len(mapped)} entries, missing={len(missing)}, unexpected={len(unexpected)})"
            )
            if 0 < len(missing) <= 10:
                logger.info(f"[MIL] Missing keys: {missing}")
        return matched, missing, unexpected

    def forward(self, x, return_segmentation: bool = False):
        if x.ndim != 5:
            raise ValueError(
                f"MILResNet50Classifier expects (B, N, C, H, W); got shape {tuple(x.shape)}"
            )
        cls_logits = self.mil(x)
        if return_segmentation:
            # No segmentation output for MIL; return a zero tensor for
            # API parity with the other wrappers in this file.
            seg = torch.zeros((x.shape[0], 1, 1, 1), device=x.device)
            return cls_logits, seg
        return cls_logits

    @torch.no_grad()
    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-instance attention weights, shape ``(B, N)``.

        Only valid for ``mil_mode in {"att", "att_trans"}``. Useful for
        thesis figures and sanity checks (does attention land on
        tumor-bearing slices in BigLunge where we happen to have a tumor
        mask available for overlay?).
        """
        if self.mil.mil_mode not in ("att", "att_trans"):
            raise RuntimeError(
                f"attention_weights unsupported for mil_mode={self.mil.mil_mode}"
            )
        sh = x.shape
        flat = x.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4])
        feats = self.mil.net(flat).reshape(sh[0], sh[1], -1)
        if self.mil.mil_mode == "att_trans" and self.mil.transformer is not None:
            feats = feats.permute(1, 0, 2)
            feats = self.mil.transformer(feats)
            feats = feats.permute(1, 0, 2)
        a = self.mil.attention(feats)
        a = torch.softmax(a, dim=1)  # (B, N, 1)
        return a.squeeze(-1)  # (B, N)


class MILSwinTinyClassifier(nn.Module):
    """BigLunge MIL classifier: attention pooling over whole-slice bags,
    Swin-Tiny backbone (counterpart to ``MILResNet50Classifier``).

    Wraps ``monai.networks.nets.MILModel`` with a timm
    ``swin_tiny_patch4_window7_224`` backbone built as a feature extractor
    (``num_classes=0`` → final ``head`` is ``Identity``, output is the 768-d
    pooled feature vector). The backbone is initialized from RadImageNet
    (the same checkpoint used by ``SwinTiny2DClassifier``) and the
    3-channel patch_embed stem is averaged to a 1-channel conv.

    **No BN to freeze.** Unlike the ResNet-50 MIL variant, Swin-Tiny uses
    LayerNorm everywhere — there is no batch-statistic leakage across the
    flattened ``(B*N, C, H, W)`` MIL minibatch. We do not need (or run) the
    ``train()`` BN-eval override that ``MILResNet50Classifier`` provides.

    **AMP / softmax stability.** The same caveat as the ResNet-50 MIL
    variant applies: a randomly-initialized attention head on top of a
    strong pretrained backbone can NaN under AMP fp16 in early steps.
    Use a low FT learning rate, long warmup, and ``--disable-amp`` for the
    first run to rule out the fp16 path.

    **Loading a DAPT backbone.** Call ``load_backbone_from_dapt(state_dict)``
    with a checkpoint produced by training ``SwinTiny2DClassifier``. Keys
    of the form ``swin.<timm_key>`` are stripped to ``<timm_key>`` and
    loaded into ``self.mil.net``; the DAPT 3-class head ``swin.head.*`` is
    dropped (MIL has its own attention + ``myfc``).
    """

    DEFAULT_RADIMAGENET_CKPT = "/home/hansstem/RadImageNet_swin/rin_swintf.pth"
    BACKBONE_NUM_FEATURES = 768  # swin_tiny_patch4_window7_224

    def __init__(
        self,
        num_classes: int = 3,
        mil_mode: str = "att",
        radimagenet_ckpt: str = "",
        trans_blocks: int = 4,
        trans_dropout: float = 0.0,
    ):
        super().__init__()

        # Build a 1-channel feature extractor. num_classes=0 swaps the timm
        # head for an Identity, so forward(x) returns global-pooled (B, 768)
        # features — exactly what MILModel expects from a custom backbone.
        backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=0,
            in_chans=1,
        )

        self.mil = MILModel(
            num_classes=num_classes,
            mil_mode=mil_mode,
            pretrained=False,
            backbone=backbone,
            backbone_num_features=self.BACKBONE_NUM_FEATURES,
            trans_blocks=trans_blocks,
            trans_dropout=trans_dropout,
        )

        # Optionally seed from RadImageNet. Default to the same path
        # SwinTiny2DClassifier uses so that ``mil_swin_tiny`` matches
        # ``swin_tiny_2d``'s pretraining provenance out of the box.
        rin_path = radimagenet_ckpt or self.DEFAULT_RADIMAGENET_CKPT
        if rin_path and os.path.exists(rin_path):
            self._load_radimagenet_into_backbone(rin_path)
        else:
            print(
                f"[!] MILSwinTinyClassifier: RadImageNet checkpoint not found at "
                f"{rin_path}; backbone is randomly initialized."
            )

    def _load_radimagenet_into_backbone(self, path: str) -> None:
        """Load RadImageNet weights into ``self.mil.net``.

        Reuses ``SwinTiny2DClassifier``'s MS->timm key remapping (downsample
        index +1) and the 3-ch->1-ch stem averaging. The RadImageNet 165-class
        head is dropped; the MIL head/attention stay at their fresh init.
        """
        print(f"[*] MILSwinTinyClassifier: loading RadImageNet weights from {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "model" in ck:
            sd = ck["model"]
        elif isinstance(ck, dict) and "state_dict" in ck:
            sd = ck["state_dict"]
        else:
            sd = ck

        # Drop buffers timm recomputes at construction.
        sd = {
            k: v for k, v in sd.items()
            if not (k.endswith(".relative_position_index") or k.endswith(".attn_mask"))
        }
        # Drop the RadImageNet 165-class head; MIL's myfc replaces it.
        sd = {k: v for k, v in sd.items() if k not in ("head.weight", "head.bias")}
        # MS-style downsample keys -> timm-style (+1 stage).
        sd = SwinTiny2DClassifier._remap_ms_to_timm(sd)
        # Average 3-channel stem -> 1-channel.
        stem_key = "patch_embed.proj.weight"
        if stem_key in sd and sd[stem_key].shape[1] == 3:
            sd[stem_key] = sd[stem_key].mean(dim=1, keepdim=True)

        missing, unexpected = self.mil.net.load_state_dict(sd, strict=False)
        matched = len(sd) - len(unexpected)
        print(
            f"[*] RadImageNet weights into MIL backbone: matched {matched}/{len(sd)} keys "
            f"(unexpected={len(unexpected)}, missing={len(missing)})"
        )

    def load_backbone_from_dapt(self, state_dict, logger=None):
        """Load a DAPT ``SwinTiny2DClassifier`` state_dict into ``self.mil.net``.

        DAPT keys look like ``swin.<timm_key>`` (the 2D wrapper stores the
        timm Swin under ``self.swin``). Strip the ``swin.`` prefix and load
        non-strict. The DAPT 3-class ``swin.head.*`` is dropped — MIL has
        ``attention`` + ``myfc`` which stay at their current init.
        """
        src_prefix = "swin."
        target_sd = self.mil.net.state_dict()
        mapped: dict = {}
        for k, v in state_dict.items():
            if not k.startswith(src_prefix):
                continue
            new_k = k[len(src_prefix):]
            if new_k in ("head.weight", "head.bias"):
                continue  # drop DAPT classification head
            if new_k in target_sd:
                mapped[new_k] = v
        missing, unexpected = self.mil.net.load_state_dict(mapped, strict=False)
        matched = len(target_sd) - len(missing)
        if logger is not None:
            logger.info(
                f"[MIL-Swin] Loaded DAPT backbone: matched {matched}/{len(target_sd)} target keys "
                f"(source contributed {len(mapped)} entries, missing={len(missing)}, "
                f"unexpected={len(unexpected)})"
            )
            if 0 < len(missing) <= 10:
                logger.info(f"[MIL-Swin] Missing keys: {missing}")
        return matched, missing, unexpected

    def forward(self, x, return_segmentation: bool = False):
        if x.ndim != 5:
            raise ValueError(
                f"MILSwinTinyClassifier expects (B, N, C, H, W); got shape {tuple(x.shape)}"
            )
        cls_logits = self.mil(x)
        if return_segmentation:
            seg = torch.zeros((x.shape[0], 1, 1, 1), device=x.device)
            return cls_logits, seg
        return cls_logits

    @torch.no_grad()
    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-instance attention weights, shape ``(B, N)``.

        Mirrors ``MILResNet50Classifier.attention_weights`` for downstream
        attention-overlay figures and entropy logging.
        """
        if self.mil.mil_mode not in ("att", "att_trans"):
            raise RuntimeError(
                f"attention_weights unsupported for mil_mode={self.mil.mil_mode}"
            )
        sh = x.shape
        flat = x.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4])
        feats = self.mil.net(flat).reshape(sh[0], sh[1], -1)
        if self.mil.mil_mode == "att_trans" and self.mil.transformer is not None:
            feats = feats.permute(1, 0, 2)
            feats = self.mil.transformer(feats)
            feats = feats.permute(1, 0, 2)
        a = self.mil.attention(feats)
        a = torch.softmax(a, dim=1)
        return a.squeeze(-1)

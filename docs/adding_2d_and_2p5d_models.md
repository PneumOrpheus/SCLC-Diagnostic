# Adding 2D / 2.5D models — step-by-step

You have four new model types to wire in:

1. `densenet121_2d` — 2D, 1-channel, ImageNet-pretrained (MONAI `DenseNet121`).
2. `resnet50_2d` — 2D, 1-channel, ImageNet-pretrained (MONAI `TorchVisionFCModel` wrapping torchvision `resnet50`).
3. `densenet121_2p5d` — 2.5D, `num_slices`-channel, ImageNet-pretrained (MONAI `DenseNet121`).
4. `mil_resnet50_2p5d` — 2.5D MIL: each slice is an *instance*, attention pools across slices (MONAI `MILModel`).

Models 1–3 slot into the existing 2D / 2.5D pipelines with minimal changes. Model 4 (MIL) needs a new data shape `(B, N_instances, 3, H, W)`, so it gets a dedicated transform + a third pipeline tag (`"mil"`). Everything else (DataLoader, train/validate, main.py dispatch) is reused.

Read sections 0 and 1 once — they apply to every model. Then pick models you want and implement them one at a time. After each one, run the smoke test at the bottom of its section before moving on.

---

## 0. Know the shape contract

Before touching anything, lock down what each pipeline hands the model:

| Pipeline | Sample shape out of transforms | Batch shape fed to `model.forward` |
|---|---|---|
| 3D | `(1, X, Y, Z)` | `(B, 1, X, Y, Z)` |
| 2.5D (channel-stacked) | `(num_slices, H, W)` | `(B, num_slices, H, W)` |
| 2D (single slice) | `(1, H, W)` | `(B, 1, H, W)` |
| 2.5D MIL (new) | `(num_slices, 3, H, W)` | `(B, num_slices, 3, H, W)` |

If a model's `forward` sees a shape it doesn't expect, fix the **transform or classifier wrapper**, not the training loop. The training loop is pipeline-agnostic by design.

---

## 1. Register new model types (one edit covers all four)

**File:** `model_selection.py`

At the top of the "pipeline selection" block, extend the tuples:

```python
TWO_P_FIVE_D_MODEL_TYPES = (
    "efficientnet_b0_2p5d",
    "densenet121_2p5d",
)
TWO_D_MODEL_TYPES = (
    "efficientnet_b0_2d",
    "densenet121_2d",
    "resnet50_2d",
)
MIL_MODEL_TYPES = (
    "mil_resnet50_2p5d",
)
```

Add a `get_pipeline` branch:

```python
def is_mil_model_type(model_type: str) -> bool:
    return model_type.lower() in MIL_MODEL_TYPES


def get_pipeline(model_type: str) -> str:
    if is_mil_model_type(model_type):
        return "mil"
    if is_2d_model_type(model_type):
        return "2d"
    if is_2p5d_model_type(model_type):
        return "2p5d"
    return "3d"
```

Then update the `choices=[...]` list in `main.py`'s `parse_args` to include the four new names.

---

## 2. Model 1 — `densenet121_2d`

**Why:** second-most-used 2D medical baseline after EfficientNet. Sanity check that our pipeline is backbone-agnostic.

### 2.1 Add the classifier wrapper

**File:** `model_selection.py`

```python
class DenseNet2DClassifier(nn.Module):
    """2D DenseNet121 (ImageNet-pretrained) for single-slice classification.
    ``out_channels=num_classes`` tells MONAI to build a classification head.
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        from monai.networks.nets import DenseNet121
        self.densenet = DenseNet121(
            spatial_dims=2,
            in_channels=1,
            out_channels=num_classes,
            pretrained=True,
        )

    def forward(self, x, return_segmentation=False):
        if x.ndim == 5 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        cls_logits = self.densenet(x)
        if return_segmentation:
            seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            return cls_logits, seg_logits
        return cls_logits
```

### 2.2 Dispatch in `get_sclc_model`

Add a branch *above* the `efficientnet_b0_2d` branch (order doesn't matter, but group the 2D ones):

```python
if model_type.lower() == "densenet121_2d":
    model = DenseNet2DClassifier(num_classes=3)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    return model
```

### 2.3 Smoke test

```bash
python -c "
import torch
from model_selection import get_sclc_model
m = get_sclc_model('', 'densenet121_2d')
print(m(torch.randn(2,1,224,224)).shape)  # expect (2, 3)
"
```

### 2.4 Run

```bash
python main.py --mode full --model-type densenet121_2d --testing
```

No other code changes required — the 2D pipeline already feeds `(B, 1, H, W)`.

---

## 3. Model 2 — `resnet50_2d` (via `TorchVisionFCModel`)

**Why:** ResNet-50 is the expected baseline in every classification paper. MONAI's `TorchVisionFCModel` handles channel adaptation and head replacement for you.

### 3.1 Add the classifier wrapper

**File:** `model_selection.py`

```python
class TorchVisionResNet2DClassifier(nn.Module):
    """2D ResNet-50 (torchvision ImageNet weights) wired via MONAI's
    TorchVisionFCModel. The stem conv is adapted to ``in_channels=1`` and the
    final FC is replaced with a 3-class head.
    """
    def __init__(self, num_classes: int = 3, model_name: str = "resnet50"):
        super().__init__()
        from monai.networks.nets import TorchVisionFCModel
        self.backbone = TorchVisionFCModel(
            model_name=model_name,
            num_classes=num_classes,
            dim=2,
            in_channels=1,
            pretrained=True,
            pool=None,        # torchvision resnet already has avgpool before fc
            use_conv=False,
        )

    def forward(self, x, return_segmentation=False):
        if x.ndim == 5 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        cls_logits = self.backbone(x)
        if return_segmentation:
            seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            return cls_logits, seg_logits
        return cls_logits
```

### 3.2 Dispatch

```python
if model_type.lower() == "resnet50_2d":
    model = TorchVisionResNet2DClassifier(num_classes=3, model_name="resnet50")
    # checkpoint loading same pattern as above
    return model
```

### 3.3 Important gotcha

`TorchVisionFCModel` with `in_channels=1` re-initializes the stem conv (loses ImageNet weights on that layer). If you want to *preserve* the ImageNet stem, feed 3 channels instead by replicating the grayscale slice. Option: add a `Lambdad(keys=["image"], func=lambda x: x.repeat(3, 1, 1))` at the end of `_build_2d_pipeline` and set `in_channels=3`. Document whichever you pick — it's a real ablation choice, not a detail.

### 3.4 Smoke test + run

Same pattern as section 2.

---

## 4. Model 3 — `densenet121_2p5d`

**Why:** matches the 2.5D EfficientNet comparison but with dense feature reuse.

### 4.1 Add the classifier wrapper

**File:** `model_selection.py`

```python
class DenseNet2p5DClassifier(nn.Module):
    """2.5D DenseNet121: ``num_slices`` axial slices as input channels."""
    def __init__(self, num_slices: int = 5, num_classes: int = 3):
        super().__init__()
        from monai.networks.nets import DenseNet121
        self.num_slices = int(num_slices)
        self.densenet = DenseNet121(
            spatial_dims=2,
            in_channels=self.num_slices,
            out_channels=num_classes,
            pretrained=True,
        )

    def forward(self, x, return_segmentation=False):
        if x.ndim == 5 and x.shape[1] == 1:
            x = x.squeeze(1)
        cls_logits = self.densenet(x)
        if return_segmentation:
            seg_logits = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=x.device)
            return cls_logits, seg_logits
        return cls_logits
```

### 4.2 Dispatch

```python
if model_type.lower() == "densenet121_2p5d":
    model = DenseNet2p5DClassifier(num_slices=num_slices, num_classes=3)
    # standard checkpoint loading
    return model
```

### 4.3 Smoke test

```python
m = get_sclc_model('', 'densenet121_2p5d', num_slices=5)
m(torch.randn(2, 5, 96, 96)).shape  # (2, 3)
```

### 4.4 Run

```bash
python main.py --mode full --model-type densenet121_2p5d --testing
```

No data / transform / training changes required.

---

## 5. Model 4 — `mil_resnet50_2p5d` (the interesting one)

This needs a new data shape (`(num_slices, 3, H, W)` per sample) and therefore a new transform + pipeline tag. The training/validation loops from the 2.5D path work unchanged as long as the collate emits a 4-D tensor per sample and the classifier accepts the resulting 5-D batch.

### 5.1 Add the instance-axis transform

**File:** `data/transforms.py`

```python
class AxialSlicesAsInstancesd(MapTransform):
    """Convert (C=1, X, Y, Z) -> (Z, 3, X, Y): each axial slice becomes an
    instance with a replicated 3-channel image (for ImageNet-pretrained
    backbones). Drops the original single-channel axis.
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            vol = d[key]
            # (1, X, Y, Z) -> (Z, X, Y)
            if isinstance(vol, torch.Tensor):
                if vol.ndim == 4:
                    vol = vol[0]
                vol = vol.permute(2, 0, 1).contiguous()      # (Z, X, Y)
                vol = vol.unsqueeze(1).expand(-1, 3, -1, -1) # (Z, 3, X, Y)
                d[key] = vol.contiguous()
            else:
                if vol.ndim == 4:
                    vol = vol[0]
                vol = np.transpose(vol, (2, 0, 1))              # (Z, X, Y)
                vol = np.broadcast_to(vol[:, None], (vol.shape[0], 3, vol.shape[1], vol.shape[2]))
                d[key] = np.ascontiguousarray(vol)
        return d
```

### 5.2 Add a MIL transform builder

Reuse `_build_25d_pipeline` but swap `AxialSlicesAsChannelsd` for `AxialSlicesAsInstancesd`. Simplest:

```python
def _build_mil_pipeline(img_size: int, num_slices: int, train: bool) -> list:
    pipeline = _build_25d_pipeline(img_size=img_size, num_slices=num_slices, train=train)
    # Swap the final slice-as-channel step for slice-as-instance
    new = []
    for t in pipeline:
        if isinstance(t, AxialSlicesAsChannelsd):
            new.append(AxialSlicesAsInstancesd(keys=["image"]))
        else:
            new.append(t)
    return new


def get_train_transforms_mil(img_size: int = 96, num_slices: int = 5) -> Compose:
    return Compose(_build_mil_pipeline(img_size, num_slices, train=True))


def get_val_transforms_mil(img_size: int = 96, num_slices: int = 5) -> Compose:
    return Compose(_build_mil_pipeline(img_size, num_slices, train=False))
```

### 5.3 Add a MIL dataset builder

**File:** `data/data_loader.py`

Simplest path: reuse everything in `create_dataset_2p5d`, parameterized on which transforms to use. Either:

- (a) **Add an argument** `transforms_fn: tuple = (get_train_transforms_2p5d, get_val_transforms_2p5d)` to `create_dataset_2p5d`, and pass the MIL pair when building a MIL dataset; or
- (b) **Copy the function** into `create_dataset_mil` and change only the transforms + `cache_name` (`monai_biglunge_mil` / `monai_lung_pet_ct_clean_mil`). 

Option (b) is uglier but lower-risk — pick it if you'd rather not touch `create_dataset_2p5d`. Option (a) is cleaner if you plan to add a fourth pipeline later.

Cache name **must** differ from 2.5D's so the `PersistentDataset` doesn't reuse stale `(num_slices, H, W)` cache entries as if they were `(num_slices, 3, H, W)`.

### 5.4 Add the classifier wrapper

**File:** `model_selection.py`

```python
class MIL2p5DClassifier(nn.Module):
    """Attention-MIL over axial slices. Each slice is one instance; a shared
    ImageNet-pretrained backbone encodes it, and a transformer-attention head
    pools instance features into one bag (patient) prediction.
    """
    def __init__(self, num_classes: int = 3, backbone: str = "resnet50", mil_mode: str = "att_trans"):
        super().__init__()
        from monai.networks.nets import MILModel
        # MILModel expects (B, N_instances, 3, H, W). No ``spatial_dims`` arg —
        # the backbone is fixed 2D. Channel count is 3 (ImageNet stem preserved).
        self.mil = MILModel(
            num_classes=num_classes,
            mil_mode=mil_mode,
            pretrained=True,
            backbone=backbone,
        )

    def forward(self, x, return_segmentation=False):
        # x is (B, N, 3, H, W)
        cls_logits = self.mil(x)
        if return_segmentation:
            # Spatial dims are per-instance H/W — return a zero placeholder so
            # the shared training loop's +seg branch doesn't crash if called.
            seg_logits = torch.zeros((x.shape[0], 1, x.shape[-2], x.shape[-1]), device=x.device)
            return cls_logits, seg_logits
        return cls_logits
```

`mil_mode` options per MONAI: `"mean"`, `"max"`, `"att"`, `"att_trans"`, `"att_trans_pyramid"`. `"att_trans"` (attention + transformer pooling) is the most expressive and is what most MIL-SCLC papers use. Worth reporting an ablation over `mil_mode` in the thesis — cheap to run.

### 5.5 Dispatch

```python
if model_type.lower() == "mil_resnet50_2p5d":
    model = MIL2p5DClassifier(num_classes=3, backbone="resnet50", mil_mode="att_trans")
    # standard checkpoint loading
    return model
```

### 5.6 Wire up the pipeline in `main.py`

**File:** `main.py`, inside `create_dataloaders`:

```python
if pipeline == "mil":
    train_ds, val_ds, test_ds = create_dataset_mil(
        data_path=data_path,
        csv_path=csv_path,
        dataset_type=dataset_type,
        img_size=args.img_size_2p5d,
        num_slices=args.num_slices,
        tumor_mask_suffix=args.tumor_mask_suffix,
        testing=args.testing,
    )
    collate_fn = simple_collate_fn   # the default collate stacks (N,3,H,W) into (B,N,3,H,W) fine
elif pipeline == "2d":
    ...
```

The existing `simple_collate_fn` handles this because `torch.stack` over a list of `(N, 3, H, W)` tensors yields `(B, N, 3, H, W)` — no special collate needed. Sanity-check this after step 5.7.

In `main()`'s train/validate dispatch, MIL uses the same loops as 2.5D/3D:

```python
if pipeline == "2d":
    train_fn, validate_fn = train_epoch_2d, validate_epoch_2d
else:
    train_fn, validate_fn = train_epoch, validate_epoch
```

(No change — already handles MIL because it falls into the `else` branch.)

### 5.7 Smoke test

```python
import torch
from model_selection import get_sclc_model
m = get_sclc_model('', 'mil_resnet50_2p5d')
# (B=2, N_instances=5, C=3, H=96, W=96)
print(m(torch.randn(2, 5, 3, 96, 96)).shape)   # expect (2, 3)
```

Then a full transform-through-collate test on a tiny fake volume — same pattern as the 2D smoke test we already ran, but with `get_val_transforms_mil` and checking final shape `(N, 3, H, W)`.

### 5.8 Memory warning

MIL is `num_slices`× more expensive per sample than channel-stacked 2.5D (each slice is a full forward pass through ResNet-50). At `num_slices=5, batch_size=2, img_size=96`, a single ResNet-50 forward is ~1.2GB activations — MIL batch is ~12GB. Start with `--batch-size 1 --accumulation-steps 8` and increase `num_slices` cautiously.

### 5.9 Run

```bash
python main.py --mode full --model-type mil_resnet50_2p5d --batch-size 1 --accumulation-steps 8 --testing
```

---

## 6. End-to-end validation checklist

Before reporting numbers, for *each* new model:

1. **Parameter count logs correctly** — look at the "Initialized X Classifier. Total Params: N" line; compare to expected (DenseNet121 ≈ 7M, ResNet50 ≈ 23.5M, MIL-ResNet50 ≈ 24.5M with transformer head).
2. **First training epoch loss decreases** — if it doesn't, the stem init (for 1-channel models) likely broke pretrained weights. Check with a frozen-backbone probe for 1 epoch.
3. **Val confusion matrix is non-trivial on `--testing`** — if every prediction is class 0, your head isn't learning; inspect the loss.
4. **Cache namespaces are disjoint** — list `~/.cache/monai_*` after the first run and confirm no shared directory across pipelines.

---

## 7. Suggested thesis comparison table

Aim for one row per (pipeline, architecture) cell so reviewers can read it as a 2×3 grid:

| Pipeline \\ Arch | EfficientNet-B0 | DenseNet121 | ResNet50 |
|---|---|---|---|
| 2D | ✅ (have) | add (§2) | add (§3) |
| 2.5D stacked | ✅ (have) | add (§4) | — |
| 2.5D MIL | — | — | add (§5) |

That's 3 new cells + 1 methodology contrast (MIL). Keeps the thesis tractable and the story clean: "architecture matters less than the dimensional prior (2D vs 2.5D), and explicit slice-attention (MIL) does / doesn't beat channel-mixing."

---

## 8. Things I deliberately skipped and why

- **SEResNet50/SEResNext50:** MONAI's SE nets hard-code `in_channels=3` in the stem. Adapting them to 1-channel needs a manual first-conv replacement and re-init of one conv, which gains you nothing over `TorchVisionFCModel("resnet50")`. Add only if a reviewer asks.
- **FlexibleUNet:** It's a segmentation net. Useful only if you want a 2D equivalent of the SwinUNETR seg-aux-loss trick. Orthogonal to your current plan.
- **ViT in 2D:** MONAI's `ViT` has no pretrained weights, and you already argued in `model_selection.py` that training ViT from scratch on 300 volumes doesn't work. Same argument at 2D — skip.
- **MONAI Model Zoo lung/CT models:** none of them are 2D classifiers. `lung_nodule_ct_detection` is a 3D RetinaNet; useful as a future "pretrained CT backbone" experiment, not for this benchmark.

---

## 9. Order to implement

1. **DenseNet121 2.5D** (§4) — smallest change, highest confidence the pipeline is backbone-agnostic.
2. **DenseNet121 2D** (§2) — same pattern, different pipeline, confirms 2D path is backbone-agnostic too.
3. **ResNet50 2D** (§3) — tests the `TorchVisionFCModel` path; if it works, any torchvision backbone is one line away.
4. **MIL-ResNet50 2.5D** (§5) — longest implementation, do it last when everything else is de-risked.

Each of the first three is ~30 lines of code and ~10 minutes. MIL is ~100 lines and the first training run will probably need a batch-size / LR sweep.

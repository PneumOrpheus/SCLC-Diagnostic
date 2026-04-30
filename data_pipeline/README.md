# `data_pipeline/` — one-shot dataset acquisition + preprocessing

Code in this directory produces the inputs the training package consumes. It runs **once per dataset**, not on every training invocation, which is why it lives outside the `sclc/` package. The active runtime data loading code is under `sclc/data/`.

## What goes where

| File | Purpose |
|---|---|
| `fetch_tcia.ipynb` | Download Lung-PET-CT-Dx via the TCIA API. |
| `recover_annotations.py` | Re-pull missing XML annotations from TCIA when a fetch was incomplete. |
| `annotation_mapping.py` | Helper for aligning DICOM series with the per-slice XML annotations. |
| `create_masks.py` | Convert DICOM + XML annotations into NIfTI volumes + `_mask.nii.gz` sidecars. Output goes to `/home/data/Lung-PET-CT-Dx-Clean`, which is what `sclc.main` reads at training time. Binds masks to SOPInstanceUID, so each patient gets only the reconstruction the radiologist annotated (typically 5 mm). |
| `upgrade_thin_reconstructions.py` | For patients where a thinner CT reconstruction (≤ 1.25 mm) of the same study is available in DICOM but did not get a mask through `create_masks.py` (the thin reconstruction has different SOPInstanceUIDs than the annotated thick one), convert the thin DICOMs to NIfTI and spatially resample the existing thick mask onto the thin grid. Multi-phase studies are handled by picking the AcquisitionTime sub-volume whose Z-range overlaps the mask. Run once after `create_masks.py`; idempotent. |
| `notebooks/biglunge_audit.ipynb` | BigLunge tumor-mask audit. Source for the `min_tumor_pixels` threshold and the truncated-lung-mask exclusion list (`sclc/data/exclusions.py`). |
| `notebooks/eda_2d.ipynb` | 2D-pipeline exploratory data analysis. Produces several thesis figures (under `notebooks/output/`). |

## Reproducibility chain

1. `fetch_tcia.ipynb` → raw DICOM + XML annotations
2. `recover_annotations.py` (only if step 1 had gaps)
3. `create_masks.py` → `/home/data/Lung-PET-CT-Dx-Clean/{patient}/{series}_image.nii.gz` + `_mask.nii.gz`
4. `upgrade_thin_reconstructions.py` → adds a thin (≤ 1.25 mm) reconstruction next to the thick one for patients who have one in DICOM. Loader (`sclc/data/loaders.py`) sorts patient scans by Z-spacing, so the thin reconstruction is preferred when `max_scans_per_patient` caps.
5. BigLunge data is provided externally (SINTEF) — no acquisition step in this repo.
6. `sclc.main --config configs/experiments/<model>.yaml` consumes both datasets.

The training pipeline does **not** import anything from `data_pipeline/`. If you need to re-curate the dataset, run the relevant script from this directory; otherwise the code here can be ignored.

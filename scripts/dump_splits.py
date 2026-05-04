"""Dump patient-level train/val/test splits for both datasets to a JSON file.

Reads the same loaders the training pipeline uses (with the standard
seed=42, val_frac=0.15, test_frac=0.15) so the dump matches what the
models actually saw.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from sclc.data.loaders import (
    CLASS_NAMES,
    get_biglunge_data_list,
    get_lung_pet_ct_dx_data_list,
)

LPCD_DATA = "/home/data/Lung-PET-CT-Dx-Clean"
BIGLUNGE_DATA = "/home/data/TrainingData"
BIGLUNGE_CSV = "/home/data/TrainingData/patients_parameters.csv"
OUT_PATH = Path(__file__).resolve().parents[1] / "results" / "splits.json"


def _patient_index(splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """{split: [{patient_id, class_idx, class_name}]} keeping unique patients only."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for split, entries in splits.items():
        seen: Dict[str, Dict[str, Any]] = {}
        for e in entries:
            # Lung-PET-CT-Dx entries don't carry patient_id; derive from image path.
            pid = e.get("patient_id")
            if not pid:
                img = e.get("image", "")
                pid = Path(img).parent.name if img else None
            if not pid or pid in seen:
                continue
            label = int(e.get("scan_label", -1))
            seen[pid] = {
                "patient_id": pid,
                "class_idx": label,
                "class_name": CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else "?",
            }
        out[split] = sorted(seen.values(), key=lambda r: r["patient_id"])
    return out


def main() -> None:
    print(f"[splits] Loading Lung-PET-CT-Dx from {LPCD_DATA}")
    lpcd = get_lung_pet_ct_dx_data_list(data_path=LPCD_DATA)
    print(f"[splits] Loading BigLunge from {BIGLUNGE_DATA}")
    biglunge = get_biglunge_data_list(data_path=BIGLUNGE_DATA, csv_path=BIGLUNGE_CSV)

    out: Dict[str, Any] = {
        "params": {"seed": 42, "val_frac": 0.15, "test_frac": 0.15},
        "class_names": CLASS_NAMES,
        "lung_pet_ct_dx": _patient_index(lpcd),
        "biglunge": _patient_index(biglunge),
    }
    for ds_name, ds in (("lung_pet_ct_dx", out["lung_pet_ct_dx"]),
                        ("biglunge", out["biglunge"])):
        n = {k: len(v) for k, v in ds.items()}
        print(f"[splits] {ds_name}: {n}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[splits] Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

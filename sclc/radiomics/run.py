"""End-to-end driver for the radiomics pipeline.

Steps (each step skipped if its inputs are already on disk):

  1. Phase 1: extract baseline features for both datasets.
  2. Phase 1b: extract dilated + eroded features for both datasets (Phase 2 input).
  3. Phase 2: stability filter (ICC + NZV + correlation).
  4. Phase 3: train + eval all model_types.
  5. Phase 4: SHAP interpretation.
  6. Phase 5: write per-model _provenance.json under
     ``results/thesis/2d/per_model/<model_type>/`` and patch
     ``scripts/build_thesis_results.py``'s PIPELINES dict so headline rebuild
     picks up the new rows.

Designed to be re-runnable end-to-end. Use ``--force`` to skip the
"input already exists" checks.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from sclc.radiomics import extract, stability, train_eval, interpret

REPO_ROOT = Path(__file__).resolve().parents[2]
RADIOMICS_DIR = REPO_ROOT / "results" / "radiomics"
PER_MODEL_DIR = REPO_ROOT / "results" / "thesis" / "2d" / "per_model"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_thesis_results.py"


# ---------- per-model provenance ---------------------------------------------

def _provenance_for(
    model_type: str,
    algo: str,
    summary: Dict[str, Any],
    interp: Dict[str, Any],
) -> Dict[str, Any]:
    # Find this model_type's report block in the summary.
    report = next(
        (r for r in summary.get("model_reports", []) if r["model_type"] == model_type),
        None,
    )
    if report is None:
        return {}

    output_dir = REPO_ROOT / "results" / "output" / "2d" / model_type
    probs = {phase: info["probs_json"] for phase, info in report.get("evals", {}).items()}

    return {
        "model_type": model_type,
        "pipeline": "2d",
        "model_family": "radiomics",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "algo": algo,
        "trained_on": report["trained_on"],
        "is_winner_for_dataset": report.get("is_winner_for_dataset", False),
        "hyperparams": report["hyperparams"],
        "n_features_selected": report["n_features_selected"],
        "selected_features": report["selected_features"],
        "extractor_settings": {
            "binWidth": 25,
            "resampledPixelSpacing": [1.0, 1.0, 1.0],
            "feature_classes": ["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"],
        },
        "stability_filter": {
            "icc_cutoff": 0.75,
            "nzv_std_cutoff": 1e-6,
            "corr_cutoff": 0.9,
        },
        "smote": True,
        "feature_selection": {
            "method": "LASSO multinomial L1-LR",
            "cap_rule": "sqrt(N_train)",
        },
        "metrics_jsonl_source": str(output_dir / "metrics.jsonl"),
        "inference_probs_sources": probs,
        "shap_artifacts": interp.get(model_type, {}),
        "checkpoints": {
            # Radiomics models aren't .pth files, but keep the key so the
            # provenance schema stays compatible with the deep models.
            "lasso_selected_features": report["selected_features"],
        },
    }


def _write_provenances(summary: Dict[str, Any], interp: Dict[str, Any]) -> List[str]:
    written: List[str] = []
    for algo in ("svm", "rf", "gb"):
        for model_type in (f"radiomics_{algo}", f"radiomics_{algo}_bl"):
            prov = _provenance_for(model_type, algo, summary, interp)
            if not prov:
                continue
            out_dir = PER_MODEL_DIR / model_type
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "_provenance.json"
            with open(out_path, "w") as f:
                json.dump(prov, f, indent=2, default=str)
            written.append(str(out_path))
    return written


# ---------- patch build_thesis_results PIPELINES + MODEL_LABEL ----------------

NEW_MODEL_TYPES = (
    "radiomics_svm", "radiomics_svm_bl",
    "radiomics_rf",  "radiomics_rf_bl",
    "radiomics_gb",  "radiomics_gb_bl",
)
MODEL_LABELS_RADIOMICS = {
    "radiomics_svm":    "Radiomics SVM (LPCT-train → LPCT-test / BL-test transfer)",
    "radiomics_svm_bl": "Radiomics SVM (BL-train → BL-test in-sample)",
    "radiomics_rf":     "Radiomics RF (LPCT-train → LPCT-test / BL-test transfer)",
    "radiomics_rf_bl":  "Radiomics RF (BL-train → BL-test in-sample)",
    "radiomics_gb":     "Radiomics GB (LPCT-train → LPCT-test / BL-test transfer)",
    "radiomics_gb_bl":  "Radiomics GB (BL-train → BL-test in-sample)",
}


def _patch_build_thesis_results() -> None:
    """Idempotently add the radiomics model_types to PIPELINES['2d'] and
    MODEL_LABEL in scripts/build_thesis_results.py.
    """
    src = BUILD_SCRIPT.read_text()
    if "radiomics_svm" in src:
        print("[run] build_thesis_results.py already patched; skipping")
        return

    # 1. Insert radiomics models into PIPELINES["2d"].
    pipelines_marker = '"swin_tiny_2d",\n    ],'
    radiomics_block = '"swin_tiny_2d",\n        "radiomics_svm", "radiomics_svm_bl",\n        "radiomics_rf",  "radiomics_rf_bl",\n        "radiomics_gb",  "radiomics_gb_bl",\n    ],'
    if pipelines_marker not in src:
        # Be lenient about indentation drift.
        if '"swin_tiny_2d",' not in src:
            raise RuntimeError("Could not find swin_tiny_2d marker in build_thesis_results.py")
        src = src.replace(
            '"swin_tiny_2d",\n    ],',
            radiomics_block,
            1,
        )
    else:
        src = src.replace(pipelines_marker, radiomics_block, 1)

    # 2. Insert MODEL_LABEL entries.
    label_marker = '"swin_unetr":          "SwinUNETR (3D)",\n}'
    label_addition = '"swin_unetr":          "SwinUNETR (3D)",\n'
    for k, v in MODEL_LABELS_RADIOMICS.items():
        label_addition += f'    "{k}": {json.dumps(v)},\n'
    label_addition += "}"
    if label_marker not in src:
        raise RuntimeError("Could not find MODEL_LABEL closing marker in build_thesis_results.py")
    src = src.replace(label_marker, label_addition, 1)

    BUILD_SCRIPT.write_text(src)
    print(f"[run] patched {BUILD_SCRIPT}")


# ---------- end-to-end --------------------------------------------------------

def run_pipeline(force: bool = False, n_jobs: int = 8) -> None:
    RADIOMICS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: baseline extraction --------------------------------------
    for ds in ("lpcd", "biglunge"):
        out = RADIOMICS_DIR / f"features_{ds}.csv"
        if out.is_file() and not force:
            print(f"[run] {out} exists; skipping baseline extraction")
            continue
        extract.extract_dataset(ds, perturbation="none", n_jobs=n_jobs, out_path=out)

    # ---- Phase 1b: perturbed extractions (dilate + erode) ------------------
    for ds in ("lpcd", "biglunge"):
        for kind in ("dilate", "erode"):
            out = RADIOMICS_DIR / f"features_{ds}_{kind}.csv"
            if out.is_file() and not force:
                print(f"[run] {out} exists; skipping {kind} extraction")
                continue
            extract.extract_dataset(ds, perturbation=kind, n_jobs=n_jobs, out_path=out)

    # ---- Phase 2: stability filter ----------------------------------------
    for ds in ("lpcd", "biglunge"):
        out = RADIOMICS_DIR / f"stable_features_{ds}.csv"
        if out.is_file() and not force:
            print(f"[run] {out} exists; skipping stability filter")
            continue
        stability.run(ds)

    # ---- Phase 3: train + eval --------------------------------------------
    summary_path = RADIOMICS_DIR / "train_eval_summary.json"
    if summary_path.is_file() and not force:
        print(f"[run] {summary_path} exists; loading cached summary")
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = train_eval.run()

    # ---- Phase 4: SHAP -----------------------------------------------------
    try:
        interp = interpret.run()
    except Exception as e:
        print(f"[run] SHAP interpretation failed: {e}")
        interp = {}

    # ---- Phase 5: provenance + headline -----------------------------------
    written = _write_provenances(summary, interp)
    print(f"[run] wrote {len(written)} provenance file(s)")

    _patch_build_thesis_results()

    # Run the headline rebuild.
    print(f"[run] rebuilding 2d thesis tables...")
    py = Path("/home/hansstem/anaconda3/envs/sclc/bin/python")
    rc = subprocess.run(
        [str(py), str(BUILD_SCRIPT), "--pipeline", "2d"],
        cwd=REPO_ROOT,
        check=False,
    )
    print(f"[run] build_thesis_results returncode={rc.returncode}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="Re-run all phases even if outputs already exist.")
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()
    run_pipeline(force=args.force, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()

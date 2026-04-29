import argparse
import csv
import glob
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pydicom
from tcia_utils import nbia


def collect_local_sop_uids(dicom_root: str) -> Tuple[Set[str], Set[str]]:
    sop_uids: Set[str] = set()
    series_uids: Set[str] = set()
    for series_dir_name in os.listdir(dicom_root):
        series_dir = os.path.join(dicom_root, series_dir_name)
        if not os.path.isdir(series_dir):
            continue
        for walk_root, _, files in os.walk(series_dir):
            for fname in files:
                dcm_path = os.path.join(walk_root, fname)
                if not os.path.isfile(dcm_path):
                    continue
                try:
                    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                    sop_uids.add(str(ds.SOPInstanceUID))
                    series_uids.add(str(getattr(ds, "SeriesInstanceUID", series_dir_name)))
                except Exception:
                    continue
    return sop_uids, series_uids


def collect_missing_sops(annotation_root: str, local_sops: Set[str]) -> Dict[str, List[str]]:
    missing_by_patient: Dict[str, List[str]] = defaultdict(list)
    for patient_short in sorted(os.listdir(annotation_root)):
        patient_dir = os.path.join(annotation_root, patient_short)
        if not os.path.isdir(patient_dir):
            continue
        for xml_path in glob.glob(os.path.join(patient_dir, "*.xml")):
            sop_uid = Path(xml_path).stem
            if sop_uid not in local_sops:
                missing_by_patient[patient_short].append(sop_uid)
    return missing_by_patient


def export_missing_reports(report_dir: str, missing_by_patient: Dict[str, List[str]]) -> Tuple[str, str]:
    os.makedirs(report_dir, exist_ok=True)
    csv_path = os.path.join(report_dir, "missing_sops_by_patient.csv")
    txt_path = os.path.join(report_dir, "missing_sop_uids.txt")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["patient_short_id", "patient_id", "sop_uid"])
        for patient_short, sops in sorted(missing_by_patient.items()):
            patient_id = f"Lung_Dx-{patient_short}"
            for sop in sorted(set(sops)):
                writer.writerow([patient_short, patient_id, sop])

    with open(txt_path, "w") as f:
        for _, sops in sorted(missing_by_patient.items()):
            for sop in sorted(set(sops)):
                f.write(f"{sop}\n")

    return csv_path, txt_path


def resolve_missing_series_via_patients(
    missing_by_patient: Dict[str, List[str]],
    collection: str,
    local_series_uids: Set[str],
) -> Set[str]:
    missing_series_uids: Set[str] = set()
    for patient_short in sorted(missing_by_patient.keys()):
        patient_id = f"Lung_Dx-{patient_short}"
        try:
            series_data = nbia.getSeries(collection=collection, patientId=patient_id)
        except Exception:
            continue
        if not isinstance(series_data, list):
            continue
        for row in series_data:
            series_uid = row.get("SeriesInstanceUID")
            if not series_uid:
                continue
            if series_uid not in local_series_uids:
                missing_series_uids.add(series_uid)
    return missing_series_uids


def export_missing_series(report_dir: str, missing_series_uids: Set[str]) -> str:
    path = os.path.join(report_dir, "missing_series_uids.txt")
    with open(path, "w") as f:
        for uid in sorted(missing_series_uids):
            f.write(f"{uid}\n")
    return path


def download_missing_sops_by_patient_series_probe(
    missing_by_patient: Dict[str, List[str]],
    collection: str,
    dicom_root: str,
) -> Tuple[int, int, Dict[str, List[str]]]:
    resolved = 0
    unresolved = 0
    unresolved_by_patient: Dict[str, List[str]] = defaultdict(list)

    for patient_short, sop_list in sorted(missing_by_patient.items()):
        patient_id = f"Lung_Dx-{patient_short}"
        try:
            series_data = nbia.getSeries(collection=collection, patientId=patient_id)
        except Exception:
            for sop_uid in sorted(set(sop_list)):
                unresolved_by_patient[patient_short].append(sop_uid)
                unresolved += 1
            continue

        series_uids: List[str] = []
        if isinstance(series_data, list):
            for row in series_data:
                suid = row.get("SeriesInstanceUID")
                if suid:
                    series_uids.append(suid)

        if not series_uids:
            for sop_uid in sorted(set(sop_list)):
                unresolved_by_patient[patient_short].append(sop_uid)
                unresolved += 1
            continue

        for sop_uid in sorted(set(sop_list)):
            ok = False
            for series_uid in series_uids:
                try:
                    nbia.downloadImage(seriesUID=series_uid, sopUID=sop_uid, path=dicom_root)
                except Exception:
                    continue
                expected = os.path.join(dicom_root, series_uid, f"{sop_uid}.dcm")
                if os.path.exists(expected):
                    ok = True
                    resolved += 1
                    break

            if not ok:
                unresolved_by_patient[patient_short].append(sop_uid)
                unresolved += 1

    return resolved, unresolved, unresolved_by_patient


def export_unresolved_sops(report_dir: str, unresolved_by_patient: Dict[str, List[str]]) -> str:
    out_path = os.path.join(report_dir, "unresolved_sops_after_probe.csv")
    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["patient_short_id", "patient_id", "sop_uid"])
        for patient_short, sop_list in sorted(unresolved_by_patient.items()):
            patient_id = f"Lung_Dx-{patient_short}"
            for sop_uid in sorted(set(sop_list)):
                writer.writerow([patient_short, patient_id, sop_uid])
    return out_path


def download_missing_series(missing_series_uids: Set[str], dicom_root: str, max_workers: int = 6) -> None:
    if not missing_series_uids:
        print("No missing series to download.")
        return
    print(f"Downloading {len(missing_series_uids)} missing series...")
    nbia.downloadSeries(
        sorted(missing_series_uids),
        path=dicom_root,
        max_workers=max_workers,
    )


def rerun_mapping(
    mapper_script: str,
    dicom_root: str,
    annotation_root: str,
    nifti_root: str,
    output_root: str,
    python_exe: str,
) -> None:
    cmd = [
        python_exe,
        mapper_script,
        "--dicom-root",
        dicom_root,
        "--annotation-root",
        annotation_root,
        "--nifti-root",
        nifti_root,
        "--output-root",
        output_root,
        "--copy-unmappable",
    ]
    print("Rerun mapping command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover missing Lung-PET-CT-Dx annotation coverage by finding missing SOPs, downloading missing series, and rerunning mapping."
    )
    parser.add_argument("--dicom-root", type=str, required=True)
    parser.add_argument("--annotation-root", type=str, required=True)
    parser.add_argument("--nifti-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--mapper-script", type=str, default="/home/rhoversa/SCLC-Diagnostic/data/annotation_mapping.py")
    parser.add_argument("--report-dir", type=str, default="/home/rhoversa/SCLC-Diagnostic/output/recovery_reports")
    parser.add_argument("--collection", type=str, default="Lung-PET-CT-Dx")
    parser.add_argument("--python-exe", type=str, default="/home/rhoversa/anaconda3/bin/python")
    parser.add_argument("--download", action="store_true", help="Download missing series into dicom-root")
    parser.add_argument(
        "--probe-missing-sops",
        action="store_true",
        help="Attempt direct SOP recovery by probing downloadImage across each affected patient's series.",
    )
    parser.add_argument("--rerun-mapping", action="store_true", help="Rerun mapping after report/download")
    parser.add_argument("--max-workers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Indexing local DICOM SOP/Series UIDs...")
    local_sops, local_series = collect_local_sop_uids(args.dicom_root)
    print(f"Local SOP UIDs: {len(local_sops)}")
    print(f"Local Series UIDs: {len(local_series)}")

    print("Collecting missing SOP UIDs from annotation XML files...")
    missing_by_patient = collect_missing_sops(args.annotation_root, local_sops)
    missing_sop_total = sum(len(set(v)) for v in missing_by_patient.values())
    print(f"Patients with missing SOPs: {len(missing_by_patient)}")
    print(f"Missing SOP UIDs (unique, by patient sum): {missing_sop_total}")

    csv_path, txt_path = export_missing_reports(args.report_dir, missing_by_patient)
    print(f"Wrote report CSV: {csv_path}")
    print(f"Wrote SOP list: {txt_path}")

    print("Resolving potentially missing SeriesInstanceUIDs via affected patients...")
    missing_series = resolve_missing_series_via_patients(
        missing_by_patient=missing_by_patient,
        collection=args.collection,
        local_series_uids=local_series,
    )
    series_path = export_missing_series(args.report_dir, missing_series)
    print(f"Potentially missing series: {len(missing_series)}")
    print(f"Wrote series list: {series_path}")

    if args.download:
        download_missing_series(missing_series, args.dicom_root, max_workers=args.max_workers)

    if args.probe_missing_sops:
        print("Probing direct missing SOP downloads across affected patient series...")
        resolved, unresolved, unresolved_by_patient = download_missing_sops_by_patient_series_probe(
            missing_by_patient=missing_by_patient,
            collection=args.collection,
            dicom_root=args.dicom_root,
        )
        unresolved_path = export_unresolved_sops(args.report_dir, unresolved_by_patient)
        print(f"Direct SOP probe resolved: {resolved}")
        print(f"Direct SOP probe unresolved: {unresolved}")
        print(f"Wrote unresolved SOP report: {unresolved_path}")

    if args.rerun_mapping:
        rerun_mapping(
            mapper_script=args.mapper_script,
            dicom_root=args.dicom_root,
            annotation_root=args.annotation_root,
            nifti_root=args.nifti_root,
            output_root=args.output_root,
            python_exe=args.python_exe,
        )


if __name__ == "__main__":
    main()

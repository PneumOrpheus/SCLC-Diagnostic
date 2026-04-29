import argparse
import glob
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pydicom


@dataclass
class DicomSliceInfo:
    series_uid: str
    sop_uid: str
    row_dir_lps: np.ndarray
    col_dir_lps: np.ndarray
    ipp_lps: np.ndarray
    row_spacing_mm: float
    col_spacing_mm: float


@dataclass
class NiftiSeriesInfo:
    path: str
    affine: np.ndarray
    inv_affine: Optional[np.ndarray]
    shape: Tuple[int, ...]


def _series_uid_from_nifti_json_path(json_path: str) -> Optional[str]:
    stem = Path(json_path).stem
    match = re.match(r"^.+_((?:\d+\.)+\d+)$", stem)
    if not match:
        return None
    return match.group(1)


def _index_json_orientations(nifti_root: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    series_orientation: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for json_path in glob.glob(os.path.join(nifti_root, "*.json")):
        series_uid = _series_uid_from_nifti_json_path(json_path)
        if series_uid is None:
            continue
        try:
            with open(json_path) as f:
                meta = json.load(f)
            iop = meta.get("ImageOrientationPatientDICOM")
            if iop is None or len(iop) != 6:
                continue
            row_dir = np.asarray(iop[:3], dtype=np.float64)
            col_dir = np.asarray(iop[3:], dtype=np.float64)
            row_norm = np.linalg.norm(row_dir)
            col_norm = np.linalg.norm(col_dir)
            if row_norm == 0.0 or col_norm == 0.0:
                continue
            row_dir /= row_norm
            col_dir /= col_norm
            series_orientation[series_uid] = (row_dir, col_dir)
        except Exception:
            continue
    return series_orientation


def _lps_to_ras(points_lps: np.ndarray) -> np.ndarray:
    points_ras = points_lps.copy()
    points_ras[:, 0] *= -1.0
    points_ras[:, 1] *= -1.0
    return points_ras


def _dicom_bbox_to_patient_lps(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    slice_info: DicomSliceInfo,
) -> np.ndarray:
    corners_rc = np.array(
        [
            [ymin, xmin],
            [ymin, xmax],
            [ymax, xmin],
            [ymax, xmax],
        ],
        dtype=np.float64,
    )
    row_offsets = corners_rc[:, 0:1] * slice_info.row_spacing_mm
    col_offsets = corners_rc[:, 1:2] * slice_info.col_spacing_mm
    points = (
        slice_info.ipp_lps[None, :]
        + row_offsets * slice_info.row_dir_lps[None, :]
        + col_offsets * slice_info.col_dir_lps[None, :]
    )
    return points


def _patient_lps_to_nifti_ijk(points_lps: np.ndarray, inv_affine: np.ndarray) -> np.ndarray:
    points_ras = _lps_to_ras(points_lps)
    ones = np.ones((points_ras.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points_ras, ones], axis=1)
    vox_h = points_h @ inv_affine.T
    return vox_h[:, :3]


def _invert_affine_with_fallback(img: nib.Nifti1Image) -> Optional[np.ndarray]:
    primary = np.asarray(img.affine, dtype=np.float64)
    try:
        return np.linalg.inv(primary)
    except np.linalg.LinAlgError:
        pass

    sform, sform_code = img.get_sform(coded=True)
    if sform_code > 0:
        try:
            return np.linalg.inv(np.asarray(sform, dtype=np.float64))
        except np.linalg.LinAlgError:
            pass

    qform, qform_code = img.get_qform(coded=True)
    if qform_code > 0:
        try:
            return np.linalg.inv(np.asarray(qform, dtype=np.float64))
        except np.linalg.LinAlgError:
            pass

    return None


def _find_nifti_for_series(nifti_root: str, series_uid: str) -> Optional[str]:
    patterns = [
        os.path.join(nifti_root, f"*_{series_uid}.nii.gz"),
        os.path.join(nifti_root, f"*_{series_uid}.nii"),
        os.path.join(nifti_root, f"*_{series_uid}_*.nii.gz"),
        os.path.join(nifti_root, f"*_{series_uid}_*.nii"),
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def _parse_xml_boxes(xml_path: str) -> List[Tuple[ET.Element, float, float, float, float]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        result.append((bbox, xmin, ymin, xmax, ymax))
    return result


def _clip_bbox_to_shape(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    shape: Tuple[int, ...],
) -> Tuple[int, int, int, int]:
    max_x = max(shape[0] - 1, 0)
    max_y = max(shape[1] - 1, 0)
    xmin_i = int(np.clip(np.floor(xmin), 0, max_x))
    ymin_i = int(np.clip(np.floor(ymin), 0, max_y))
    xmax_i = int(np.clip(np.ceil(xmax), 0, max_x))
    ymax_i = int(np.clip(np.ceil(ymax), 0, max_y))
    if xmax_i < xmin_i:
        xmax_i = xmin_i
    if ymax_i < ymin_i:
        ymax_i = ymin_i
    return xmin_i, ymin_i, xmax_i, ymax_i


def _index_dicom_slices(
    dicom_root: str,
    series_orientation_map: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, DicomSliceInfo]:
    uid_to_slice: Dict[str, DicomSliceInfo] = {}
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
                    sop_uid = str(ds.SOPInstanceUID)
                    series_uid = str(getattr(ds, "SeriesInstanceUID", series_dir_name))
                    iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
                    ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
                    pixel_spacing = np.asarray(ds.PixelSpacing, dtype=np.float64)
                except Exception:
                    continue

                row_dir = iop[:3]
                col_dir = iop[3:]
                row_norm = np.linalg.norm(row_dir)
                col_norm = np.linalg.norm(col_dir)
                if row_norm == 0.0 or col_norm == 0.0:
                    continue
                row_dir /= row_norm
                col_dir /= col_norm

                if series_orientation_map is not None and series_uid in series_orientation_map:
                    json_row, json_col = series_orientation_map[series_uid]
                    row_dir = json_row
                    col_dir = json_col

                uid_to_slice[sop_uid] = DicomSliceInfo(
                    series_uid=series_uid,
                    sop_uid=sop_uid,
                    row_dir_lps=row_dir,
                    col_dir_lps=col_dir,
                    ipp_lps=ipp,
                    row_spacing_mm=float(pixel_spacing[0]),
                    col_spacing_mm=float(pixel_spacing[1]),
                )
    return uid_to_slice


def _index_nifti_series(nifti_root: str, series_uids: List[str]) -> Dict[str, NiftiSeriesInfo]:
    out: Dict[str, NiftiSeriesInfo] = {}
    for series_uid in series_uids:
        nifti_path = _find_nifti_for_series(nifti_root, series_uid)
        if nifti_path is None:
            continue
        try:
            img = nib.load(nifti_path)
        except Exception:
            continue
        inv_affine = _invert_affine_with_fallback(img)
        out[series_uid] = NiftiSeriesInfo(
            path=nifti_path,
            affine=np.asarray(img.affine, dtype=np.float64),
            inv_affine=inv_affine,
            shape=tuple(int(v) for v in img.shape),
        )
    return out


def _parse_mapped_xml_stem(stem: str) -> Optional[Tuple[str, int]]:
    match = re.match(r"^.+_((?:\d+\.)+\d+)_slice(\d+)$", stem)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def remap_from_existing_mapped_xml(
    annotation_root: str,
    nifti_root: str,
    output_root: str,
    copy_unmappable: bool = False,
) -> None:
    os.makedirs(output_root, exist_ok=True)

    series_uids: List[str] = []
    for patient_id in sorted(os.listdir(annotation_root)):
        patient_dir = os.path.join(annotation_root, patient_id)
        if not os.path.isdir(patient_dir):
            continue
        for xml_path in sorted(glob.glob(os.path.join(patient_dir, "*.xml"))):
            parsed = _parse_mapped_xml_stem(Path(xml_path).stem)
            if parsed is not None:
                series_uids.append(parsed[0])

    unique_series_uids = sorted(set(series_uids))
    nifti_index = _index_nifti_series(nifti_root, unique_series_uids)

    total_xml = 0
    mapped_xml = 0
    copied_xml = 0
    skipped_xml = 0

    for patient_id in sorted(os.listdir(annotation_root)):
        patient_in = os.path.join(annotation_root, patient_id)
        if not os.path.isdir(patient_in):
            continue

        patient_out = os.path.join(output_root, patient_id)
        os.makedirs(patient_out, exist_ok=True)

        for xml_path in sorted(glob.glob(os.path.join(patient_in, "*.xml"))):
            total_xml += 1
            parsed = _parse_mapped_xml_stem(Path(xml_path).stem)
            if parsed is None:
                if copy_unmappable:
                    shutil.copy(xml_path, os.path.join(patient_out, os.path.basename(xml_path)))
                    copied_xml += 1
                else:
                    skipped_xml += 1
                continue

            series_uid, k_idx = parsed
            nifti_info = nifti_index.get(series_uid)
            if nifti_info is None:
                if copy_unmappable:
                    shutil.copy(xml_path, os.path.join(patient_out, os.path.basename(xml_path)))
                    copied_xml += 1
                else:
                    skipped_xml += 1
                continue

            try:
                tree = ET.parse(xml_path)
                parsed_boxes = _parse_xml_boxes(xml_path)
            except Exception:
                skipped_xml += 1
                continue

            if not parsed_boxes:
                skipped_xml += 1
                continue

            for bbox_elem, xmin, ymin, xmax, ymax in parsed_boxes:
                xmin_i, ymin_i, xmax_i, ymax_i = _clip_bbox_to_shape(
                    xmin, ymin, xmax, ymax, nifti_info.shape
                )
                bbox_elem.find("xmin").text = str(xmin_i)
                bbox_elem.find("ymin").text = str(ymin_i)
                bbox_elem.find("xmax").text = str(xmax_i)
                bbox_elem.find("ymax").text = str(ymax_i)

            max_k = max(nifti_info.shape[2] - 1, 0)
            k_idx = int(np.clip(k_idx, 0, max_k))

            out_name = f"{patient_id}_{series_uid}_slice{k_idx:03d}.xml"
            out_path = os.path.join(patient_out, out_name)
            tree.write(out_path)
            mapped_xml += 1

    print("-" * 72)
    print("Annotation mapping complete (fallback mode: NIfTI + mapped XML only)")
    print("-" * 72)
    print(f"Series inferred from mapped XML names: {len(unique_series_uids)}")
    print(f"Series with NIfTI matches: {len(nifti_index)} / {len(unique_series_uids)}")
    print(f"Total XML files seen: {total_xml}")
    print(f"Validated and rewritten: {mapped_xml}")
    print(f"Copied as-is (unmappable): {copied_xml}")
    print(f"Skipped: {skipped_xml}")
    print(f"Output dir: {output_root}")


def map_annotations_with_orientation_correction(
    dicom_root: str,
    annotation_root: str,
    nifti_root: str,
    output_root: str,
    copy_unmappable: bool = False,
    fallback_from_mapped: bool = False,
) -> None:
    os.makedirs(output_root, exist_ok=True)

    series_orientation_map = _index_json_orientations(nifti_root)
    uid_to_slice = _index_dicom_slices(dicom_root, series_orientation_map=series_orientation_map)
    if not uid_to_slice:
        if fallback_from_mapped:
            print("No DICOM slices indexed. Falling back to NIfTI + existing mapped XML mode...")
            remap_from_existing_mapped_xml(
                annotation_root=annotation_root,
                nifti_root=nifti_root,
                output_root=output_root,
                copy_unmappable=copy_unmappable,
            )
            return
        raise RuntimeError(
            "No DICOM slices indexed. Check dicom_root path and DICOM contents, "
            "or rerun with --fallback-from-mapped."
        )

    series_uids = sorted({v.series_uid for v in uid_to_slice.values()})
    nifti_index = _index_nifti_series(nifti_root, series_uids)

    total_xml = 0
    mapped_xml = 0
    copied_xml = 0
    skipped_xml = 0
    missing_sop = 0
    missing_nifti_or_inv = 0
    parse_or_box_fail = 0

    for patient_id in sorted(os.listdir(annotation_root)):
        patient_in = os.path.join(annotation_root, patient_id)
        if not os.path.isdir(patient_in):
            continue

        patient_out = os.path.join(output_root, patient_id)
        os.makedirs(patient_out, exist_ok=True)

        for xml_path in sorted(glob.glob(os.path.join(patient_in, "*.xml"))):
            total_xml += 1
            sop_uid = Path(xml_path).stem

            if sop_uid not in uid_to_slice:
                missing_sop += 1
                if copy_unmappable:
                    shutil.copy(xml_path, os.path.join(patient_out, os.path.basename(xml_path)))
                    copied_xml += 1
                else:
                    skipped_xml += 1
                continue

            slice_info = uid_to_slice[sop_uid]
            nifti_info = nifti_index.get(slice_info.series_uid)
            if nifti_info is None or nifti_info.inv_affine is None:
                missing_nifti_or_inv += 1
                if copy_unmappable:
                    shutil.copy(xml_path, os.path.join(patient_out, os.path.basename(xml_path)))
                    copied_xml += 1
                else:
                    skipped_xml += 1
                continue

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                parsed_boxes = _parse_xml_boxes(xml_path)
            except Exception:
                parse_or_box_fail += 1
                skipped_xml += 1
                continue

            if not parsed_boxes:
                parse_or_box_fail += 1
                skipped_xml += 1
                continue

            k_values = []
            for bbox_elem, xmin, ymin, xmax, ymax in parsed_boxes:
                corners_lps = _dicom_bbox_to_patient_lps(xmin, ymin, xmax, ymax, slice_info)
                corners_ijk = _patient_lps_to_nifti_ijk(corners_lps, nifti_info.inv_affine)

                x_new = corners_ijk[:, 0]
                y_new = corners_ijk[:, 1]
                z_new = corners_ijk[:, 2]
                xmin_n = float(np.min(x_new))
                ymin_n = float(np.min(y_new))
                xmax_n = float(np.max(x_new))
                ymax_n = float(np.max(y_new))

                xmin_i, ymin_i, xmax_i, ymax_i = _clip_bbox_to_shape(
                    xmin_n, ymin_n, xmax_n, ymax_n, nifti_info.shape
                )

                bbox_elem.find("xmin").text = str(xmin_i)
                bbox_elem.find("ymin").text = str(ymin_i)
                bbox_elem.find("xmax").text = str(xmax_i)
                bbox_elem.find("ymax").text = str(ymax_i)

                k_values.append(float(np.mean(z_new)))

            if not k_values:
                skipped_xml += 1
                continue

            k_idx = int(np.round(np.mean(k_values)))
            max_k = max(nifti_info.shape[2] - 1, 0)
            k_idx = int(np.clip(k_idx, 0, max_k))

            out_name = f"{patient_id}_{slice_info.series_uid}_slice{k_idx:03d}.xml"
            out_path = os.path.join(patient_out, out_name)
            tree.write(out_path)
            mapped_xml += 1

    print("-" * 72)
    print("Annotation mapping complete")
    print("-" * 72)
    print(f"DICOM slices indexed: {len(uid_to_slice)}")
    print(f"Series with JSON orientation: {len(series_orientation_map)}")
    print(f"Series with NIfTI matches: {len(nifti_index)} / {len(series_uids)}")
    print(f"Total XML files seen: {total_xml}")
    print(f"Mapped with orientation correction: {mapped_xml}")
    print(f"Copied as-is (unmappable): {copied_xml}")
    print(f"Skipped: {skipped_xml}")
    print("Unmappable breakdown:")
    print(f"  missing SOP in DICOM index: {missing_sop}")
    print(f"  missing NIfTI or non-invertible affine: {missing_nifti_or_inv}")
    print(f"  XML parse/empty boxes failures: {parse_or_box_fail}")
    print(f"Output dir: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map TCIA annotation XMLs to NIfTI voxel coordinates with per-series orientation correction."
    )
    parser.add_argument("--dicom-root", type=str, required=True)
    parser.add_argument("--annotation-root", type=str, required=True)
    parser.add_argument("--nifti-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--copy-unmappable", action="store_true")
    parser.add_argument("--fallback-from-mapped", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    map_annotations_with_orientation_correction(
        dicom_root=args.dicom_root,
        annotation_root=args.annotation_root,
        nifti_root=args.nifti_root,
        output_root=args.output_root,
        copy_unmappable=args.copy_unmappable,
        fallback_from_mapped=args.fallback_from_mapped,
    )

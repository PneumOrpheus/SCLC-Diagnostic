import SimpleITK as sitk
import os
import re
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import gui
from tqdm import tqdm
import pydicom
import xml.etree.ElementTree as ET
import tempfile

DICOM_ROOT_DIR = "/home/data/Lung-PET-CT-Dx_dicom"
ANNOTATION_DIR = "/home/data/Annotation"  # Original Annotation folder matching by SOPInstanceUID
CLEAN_OUTPUT_DIR = "/home/data/Lung-PET-CT-Dx-Clean"

os.makedirs(CLEAN_OUTPUT_DIR, exist_ok=True)
print("Mapping Patient IDs to their correct XML annotations...")
patient_xmls = {}
for patient_folder in os.listdir(ANNOTATION_DIR):
    folder_path = os.path.join(ANNOTATION_DIR, patient_folder)
    if not os.path.isdir(folder_path):
        continue
    
    xml_files = glob.glob(os.path.join(folder_path, "*.xml"))
    for xml_path in xml_files:
        # The XML filename captures the precise DICOM SOPInstanceUID slice
        sop_uid = os.path.basename(xml_path).replace('.xml', '')
        if patient_folder not in patient_xmls:
            patient_xmls[patient_folder] = {}
        patient_xmls[patient_folder][sop_uid] = xml_path

print(f"Found mapped annotations for {len(patient_xmls)} patients.")

def process_series_to_mask(series_dir, patient_id, full_patient_id, series_uid, patient_xmls):
    """
    Reads a DICOM series using SimpleITK cleanly, isolating temporal/multiphasic scans 
    into separate 3D tensors. It then creates a 3D numpy array mask based on the 
    SOPInstanceUID-matched XML annotations, and writes it back inherently sharing 
    the exact physical spatial headers of the original DICOMs.
    """
    all_dcm_files = glob.glob(os.path.join(series_dir, "*.dcm"))
    if not all_dcm_files:
        return False, "No valid DICOM files found."
        
    xml_map = patient_xmls.get(patient_id, {})
    
    # Analyze DICOMs to prevent SimpleITK from flattening multiphase scans
    groups_by_time = {}
    time_with_annotations = None
    
    for f in all_dcm_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            acq_time = getattr(ds, 'AcquisitionTime', 'Unknown')
            sop_uid = str(ds.SOPInstanceUID)
            
            if acq_time not in groups_by_time:
                groups_by_time[acq_time] = []
            groups_by_time[acq_time].append(f)
            
            if sop_uid in xml_map:
                time_with_annotations = acq_time
        except Exception:
            continue
            
    # We only synthesize the specific structural phase that doctors annotated
    if time_with_annotations:
        selected_time = time_with_annotations
        sorted_times = sorted(groups_by_time.keys())
        phase_index = sorted_times.index(selected_time)
    else:
        return False, "No annotations found within this series phase."
        
    selected_files = groups_by_time[selected_time]
    
    # 2. Find the reference dcm2niix NIfTI CT image
    reference_ct_path = None
    NIFTI_ROOT = "/home/data/Lung-PET-CT-Dx"
    # Looking for PatientID_SeriesUID*.nii.gz format
    matching_niftis = glob.glob(os.path.join(NIFTI_ROOT, f"{full_patient_id}_{series_uid}*.nii*"))
    if not matching_niftis:
        return False, f"Could not find reference NIfTI CT image in {NIFTI_ROOT} to align mask."
    reference_ct_path = matching_niftis[0]

    # Load reference image and isolate 3D shape (dcm2niix often creates 4D for multiphasic)
    ct_img = sitk.ReadImage(reference_ct_path)
    if ct_img.GetDimension() == 4:
        size = list(ct_img.GetSize())
        safe_phase_index = min(phase_index, size[3] - 1)
        extractor = sitk.ExtractImageFilter()
        extractor.SetSize([size[0], size[1], size[2], 0])
        extractor.SetIndex([0, 0, 0, safe_phase_index])
        ct_img_3d = extractor.Execute(ct_img)
    else:
        ct_img_3d = ct_img
    
    # SimpleITK relies on folder-based ingestion inherently. We create a filtered temp environment.
    reader = sitk.ImageSeriesReader()
    with tempfile.TemporaryDirectory() as tmp_dir:
        for f in selected_files:
            # os.symlink is fast and minimizes disk I/O
            os.symlink(f, os.path.join(tmp_dir, os.path.basename(f)))
            
        dicom_names = reader.GetGDCMSeriesFileNames(tmp_dir)
        if not dicom_names:
            return False, "SimpleITK failed to read the filtered phase sequence."
        
        reader.SetFileNames(dicom_names)
        try:
            img_3d = reader.Execute()
        except RuntimeError as e:
            return False, f"SimpleITK failed to read series: {e}"
            
        # Get shape: SimpleITK reports size as (X, Y, Z). Numpy arrays are shaped (Z, Y, X).
        size = img_3d.GetSize()
        mask_np = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)
        has_annotation = False
        
        # SimpleITK ImageSeriesReader respects the precise Z-ordering matching the loaded img_3d
        for z, dcm_path in enumerate(dicom_names):
            try:
                ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                sop_uid = str(ds.SOPInstanceUID)
            except Exception:
                continue
                
            xml_path = xml_map.get(sop_uid)
            if xml_path:
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    
                    for obj in root.findall("object"):
                        bbox = obj.find("bndbox")
                        if bbox is not None:
                            xmin = int(float(bbox.find("xmin").text))
                            ymin = int(float(bbox.find("ymin").text))
                            xmax = int(float(bbox.find("xmax").text))
                            ymax = int(float(bbox.find("ymax").text))
                            
                            xmin = max(0, xmin)
                            ymin = max(0, ymin)
                            xmax = min(size[0], xmax)
                            ymax = min(size[1], ymax)
                            
                            mask_np[z, ymin:ymax, xmin:xmax] = 1
                            has_annotation = True
                except Exception as e:
                    print(f"Error parsing XML for {sop_uid}: {e}")
                    
        if not has_annotation:
            return False, "No annotations resolved correctly onto this phase matrix."
            
        # Re-encode numpy array back into a format SimpleITK understands
        mask_img = sitk.GetImageFromArray(mask_np)
        
        # Transfer spatial identity of raw DICOMs into the Mask.
        mask_img.CopyInformation(img_3d)
        
        # Finally, mathematically align/resample the DICOM Mask physically space to the dcm2niix array structure
        # This handles canonical Y-axis flips (and any other affine transformations) safely
        aligned_mask = sitk.Resample(
            mask_img, 
            ct_img_3d, 
            sitk.Transform(), 
            sitk.sitkNearestNeighbor, # keep mask binary (no interpolation blur)
            0.0, 
            mask_img.GetPixelID()
        )
        
        patient_out_dir = os.path.join(CLEAN_OUTPUT_DIR, full_patient_id)
        os.makedirs(patient_out_dir, exist_ok=True)
        
        mask_out_path = os.path.join(patient_out_dir, f"{series_uid}_mask.nii.gz")
        ct_out_path = os.path.join(patient_out_dir, f"{series_uid}_image.nii.gz")
        
        sitk.WriteImage(aligned_mask, mask_out_path)
        sitk.WriteImage(ct_img_3d, ct_out_path)
        
        return True, mask_out_path


if __name__ == "__main__":
    print(f"Scanning for DICOM series in {DICOM_ROOT_DIR}...")
    dicom_series_dirs = [os.path.join(DICOM_ROOT_DIR, d) for d in os.listdir(DICOM_ROOT_DIR) if os.path.isdir(os.path.join(DICOM_ROOT_DIR, d))]

    processed_count = 0
    skipped_count = 0

    for series_dir in tqdm(dicom_series_dirs, desc="Generating 3D NIfTI Masks"):
        series_uid = os.path.basename(series_dir)
        
        dicom_files = glob.glob(os.path.join(series_dir, "*.dcm"))
        if not dicom_files:
            skipped_count += 1
            continue
            
        # Get standard formatting variables to define accurate directory structures
        try:
            first_ds = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
            if not hasattr(first_ds, 'PatientID'):
                skipped_count += 1
                continue
            full_patient_id = str(first_ds.PatientID)        # e.g., 'Lung_Dx-A0001'
            patient_folder = full_patient_id.split("-")[-1]  # e.g., 'A0001' for dictionary matching
        except Exception:
            skipped_count += 1
            continue
            
        # Route execution to core block
        success, msg = process_series_to_mask(series_dir, patient_folder, full_patient_id, series_uid, patient_xmls)
        
        if success:
            processed_count += 1
        else:
            skipped_count += 1

    print(f"\n--- NIfTI Generation Complete ---")
    print(f"Verified patient masks converted and matched geometry saved: {processed_count}")
    print(f"Skipped/Empty Series: {skipped_count}")
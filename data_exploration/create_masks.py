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
DICOM_ROOT_DIR = "/home/data/Lung-PET-CT-Dx_dicom"
ANNOTATION_DIR = "/home/data/Annotation"  # Original Annotation folder matching by SOPInstanceUID
MASK_OUTPUT_DIR = "/home/data/Lung-PET-CT-Dx_masks"

os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
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
    Reads a DICOM series using SimpleITK, creates a 3D numpy array mask based on 
    the SOPInstanceUID-matched XML annotations, and writes it back natively as a 
    NIfTI volume sharing the exact spatial headers of the original DICOMs.
    """
    # Read DICOM image series exactly in SimpleITK spacing/orientation
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(series_dir)
    if not dicom_names:
        return False, "No valid DICOM series found."
    
    reader.SetFileNames(dicom_names)
    try:
        img_3d = reader.Execute()
    except RuntimeError as e:
        return False, f"SimpleITK failed to read series: {e}"
        
    # Get shape: SimpleITK reports size as (X, Y, Z). Numpy arrays are shaped (Z, Y, X).
    size = img_3d.GetSize()
    mask_np = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)
    
    # Fetch the dict of XMLs mapped to this patient
    xml_map = patient_xmls.get(patient_id, {})
    has_annotation = False
    
    # SimpleITK ImageSeriesReader respects the precise Z-ordering matching the loaded img_3d
    for z, dcm_path in enumerate(dicom_names):
        try:
            # Fast parse the header to retrieve the slice's exact SOPInstanceUID
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
                        # Parse XML boundaries (these represent X and Y coordinates on the image slice)
                        xmin = int(float(bbox.find("xmin").text))
                        ymin = int(float(bbox.find("ymin").text))
                        xmax = int(float(bbox.find("xmax").text))
                        ymax = int(float(bbox.find("ymax").text))
                        
                        # Apply boundaries preventing out-of-bounds array access
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = min(size[0], xmax)
                        ymax = min(size[1], ymax)
                        
                        # Populate mask block as 1. Numpy index order is [z, y, x]
                        mask_np[z, ymin:ymax, xmin:xmax] = 1
                        has_annotation = True
            except Exception as e:
                print(f"Error parsing XML for {sop_uid}: {e}")
                
    # Skip creating NIfTIs for unannotated scans purely designed to fill empty storage
    if not has_annotation:
        return False, "No annotations found within this series."
        
    # Re-encode numpy array back into a format SimpleITK understands
    mask_img = sitk.GetImageFromArray(mask_np)
    
    # ! CRITICAL FIX: Transfer spatial identity (Origin, Spacing, Direction Cosine Matrix).
    # Allows the resulting NIfTI mask to snap perfectly onto clinical scan spaces when superimposed
    mask_img.CopyInformation(img_3d)
    
    # Isolate folder logic exactly matching request: /home/data/Lung-PET-CT-Dx_masks/{patient_id}/{series_uid}.nii.gz
    out_dir = os.path.join(MASK_OUTPUT_DIR, full_patient_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{series_uid}.nii.gz")
    
    # Write cleanly to NIfTI format
    sitk.WriteImage(mask_img, out_path)
    
    return True, out_path
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
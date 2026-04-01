import os
import glob
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pydicom
from tqdm import tqdm

def main():
    DICOM_ROOT_DIR = "/home/data/Lung-PET-CT-Dx_dicom"
    ANNOTATION_DIR = "/home/data/Annotation"  # Original Annotation folder, NOT Z-Mapped
    OUTPUT_DIR = "/home/data/Lung-PET-CT-Dx_dicom_annot"
    UNANNOTATED_TXT_PATH = "unannotated_dicoms.txt"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Mapping Patient IDs to their correct XML annotations...")
    # 1. Map patient_id to XML annotations based on SOPInstanceUID
    patient_xmls = {}
    for patient_folder in os.listdir(ANNOTATION_DIR):
        folder_path = os.path.join(ANNOTATION_DIR, patient_folder)
        if not os.path.isdir(folder_path):
            continue
        
        xml_files = glob.glob(os.path.join(folder_path, "*.xml"))
        for xml_path in xml_files:
            sop_uid = os.path.basename(xml_path).replace('.xml', '')
            if patient_folder not in patient_xmls:
                patient_xmls[patient_folder] = {}
            patient_xmls[patient_folder][sop_uid] = xml_path

    print(f"Found annotations for {len(patient_xmls)} patients.")

    # 2. Iterate through all DICOM series
    dicom_series_dirs = [os.path.join(DICOM_ROOT_DIR, d) for d in os.listdir(DICOM_ROOT_DIR) if os.path.isdir(os.path.join(DICOM_ROOT_DIR, d))]

    unannotated_dicoms = []

    for series_dir in tqdm(dicom_series_dirs, desc="Processing DICOM Series"):
        series_uid = os.path.basename(series_dir)
        
        dicom_files = glob.glob(os.path.join(series_dir, "*.dcm"))
        if not dicom_files:
            continue
            
        # Read the first DICOM to determine the PatientID
        try:
            first_ds = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
            if not hasattr(first_ds, 'PatientID'):
                unannotated_dicoms.extend(dicom_files)
                continue
                
            full_patient_id = str(first_ds.PatientID)  # e.g., Lung_Dx-A0001
            patient_folder = full_patient_id.split("-")[-1]  # e.g., A0001
        except Exception:
            unannotated_dicoms.extend(dicom_files)
            continue
            
        slices = []
        series_has_annotation = False
        
        for fp in dicom_files:
            try:
                ds = pydicom.dcmread(fp, force=True)
                if not hasattr(ds, "pixel_array"):
                    unannotated_dicoms.append(fp)
                    continue
                    
                sop_uid = str(ds.SOPInstanceUID)
                
                xml_path = None
                if patient_folder in patient_xmls and sop_uid in patient_xmls[patient_folder]:
                    xml_path = patient_xmls[patient_folder][sop_uid]
                    series_has_annotation = True
                    
                if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
                    z_pos = float(ds.ImagePositionPatient[2])
                else:
                    z_pos = float(getattr(ds, "InstanceNumber", len(slices)))
                    
                slices.append((z_pos, ds, fp, xml_path))
            except Exception:
                unannotated_dicoms.append(fp)
                continue
                
        # Only save annotated series
        if not series_has_annotation:
            for _, _, fp, _ in slices:
                unannotated_dicoms.append(fp)
            continue
            
        slices.sort(key=lambda x: x[0])
        
        series_out_dir = os.path.join(OUTPUT_DIR, full_patient_id, series_uid)
        os.makedirs(series_out_dir, exist_ok=True)
        
        for z_pos, ds, fp, xml_path in slices:
            annotated_this_slice = False
            
            if hasattr(ds.file_meta, "TransferSyntaxUID") and ds.file_meta.TransferSyntaxUID.is_compressed:
                try:
                    ds.decompress()
                except Exception as e:
                    print(f"Failed to decompress {fp}: {e}")
                    
            if xml_path:
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    bboxes = []
                    for obj in root.findall("object"):
                        name_elem = obj.find("name")
                        bbox = obj.find("bndbox")
                        if name_elem is None or bbox is None:
                            continue
                        letter = name_elem.text.strip()
                        xmin = int(float(bbox.find("xmin").text))
                        ymin = int(float(bbox.find("ymin").text))
                        xmax = int(float(bbox.find("xmax").text))
                        ymax = int(float(bbox.find("ymax").text))
                        bboxes.append((letter, xmin, ymin, xmax, ymax))
                        
                    if bboxes:
                        pixel_array = ds.pixel_array.copy()
                        draw_value = pixel_array.max()
                        
                        for letter, xmin, ymin, xmax, ymax in bboxes:
                            xmin = max(0, xmin)
                            ymin = max(0, ymin)
                            xmax = min(pixel_array.shape[1] - 1, xmax)
                            ymax = min(pixel_array.shape[0] - 1, ymax)
                            
                            thickness = 2
                            # Horizontal
                            for t in range(thickness):
                                if ymin + t < pixel_array.shape[0]:
                                    pixel_array[ymin + t, xmin:xmax] = draw_value
                                if ymax - t >= 0:
                                    pixel_array[ymax - t, xmin:xmax] = draw_value
                            # Vertical
                            for t in range(thickness):
                                if xmin + t < pixel_array.shape[1]:
                                    pixel_array[ymin:ymax, xmin + t] = draw_value
                                if xmax - t >= 0:
                                    pixel_array[ymin:ymax, xmax - t] = draw_value
                        
                        if ds.is_little_endian != (sys.byteorder == 'little'):
                            pixel_array = pixel_array.byteswap()
                            
                        ds.PixelData = pixel_array.tobytes()
                        annotated_this_slice = True
                except Exception as e:
                    print(f"Error drawing bbox on {fp}: {e}")
            
            if not annotated_this_slice:
                unannotated_dicoms.append(fp)
                
            orig_filename = os.path.basename(fp)
            out_path = os.path.join(series_out_dir, orig_filename)
            ds.save_as(out_path)

    with open(UNANNOTATED_TXT_PATH, "w") as f:
        for dicom_path in unannotated_dicoms:
            f.write(f"{dicom_path}\n")

    print(f"Finished generating annotated DICOM images.")
    print(f"Saved {len(unannotated_dicoms)} non-annotated DICOM paths to {UNANNOTATED_TXT_PATH}")

if __name__ == "__main__":
    main()

import os
import glob
import shutil
import pydicom
from tqdm import tqdm

CLEAN_DIR = "/home/data/Lung-PET-CT-Dx-Clean"
OTHER_DIR = "/home/data/Lung-PET-CT-Dx-Other"
DICOM_DIR = "/home/data/Lung-PET-CT-Dx_dicom"

def main():
    os.makedirs(OTHER_DIR, exist_ok=True)

    if not os.path.exists(CLEAN_DIR):
        print(f"Clean directory {CLEAN_DIR} does not exist.")
        return

    patient_folders = [f for f in os.listdir(CLEAN_DIR) if os.path.isdir(os.path.join(CLEAN_DIR, f))]
    print(f"Scanning {len(patient_folders)} patients in {CLEAN_DIR} for non-CT images...")

    other_found_count = 0

    for patient_id in tqdm(patient_folders, desc="Scanning patients"):
        pat_path = os.path.join(CLEAN_DIR, patient_id)
        
        # Images are saved as {series_uid}_image.nii.gz
        image_files = glob.glob(os.path.join(pat_path, "*_image.nii*"))
        
        for img_file in image_files:
            filename = os.path.basename(img_file)
            
            # Extract the raw series_uid
            series_uid = filename.replace("_image.nii.gz", "").replace("_image.nii", "")
            
            # Look at its original DICOM Modality
            dicom_series_dir = os.path.join(DICOM_DIR, series_uid)
            is_other = False
            modality = "Unknown"
            
            if os.path.isdir(dicom_series_dir):
                dcm_files = glob.glob(os.path.join(dicom_series_dir, "*.dcm"))
                if dcm_files:
                    try:
                        # Only need to read header
                        ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
                        if hasattr(ds, 'Modality') and ds.Modality != 'CT':
                            is_other = True
                            modality = ds.Modality
                        if 'SECONDARY' in getattr(ds, 'ImageType', []) or getattr(ds, 'PhotometricInterpretation', '') == 'RGB':
                            is_other = True
                    except Exception as e:
                        print(f"Error reading DICOM for {series_uid}: {e}")
            
            if is_other:
                other_found_count += 1
                print(f"\nFound image other than CT (Modality: {modality}) for patient {patient_id}")
                
                # Setup target patient dir
                target_pat_dir = os.path.join(OTHER_DIR, patient_id)
                os.makedirs(target_pat_dir, exist_ok=True)
                
                target_img_path = os.path.join(target_pat_dir, filename)
                
                # Move the image
                shutil.move(img_file, target_img_path)
                
                # Move the corresponding mask so it doesn't get left orphaned in the Clean folder
                mask_file_gz = os.path.join(pat_path, f"{series_uid}_mask.nii.gz")
                mask_file_nii = os.path.join(pat_path, f"{series_uid}_mask.nii")
                
                if os.path.exists(mask_file_gz):
                    shutil.move(mask_file_gz, os.path.join(target_pat_dir, os.path.basename(mask_file_gz)))
                elif os.path.exists(mask_file_nii):
                    shutil.move(mask_file_nii, os.path.join(target_pat_dir, os.path.basename(mask_file_nii)))

    print("\n--- Scan Complete ---")
    print(f"Total non-CT images found and moved: {other_found_count}")

if __name__ == "__main__":
    main()

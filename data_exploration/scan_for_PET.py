import os
import glob
import shutil
import pydicom
from tqdm import tqdm

CLEAN_DIR = "/home/data/Lung-PET-CT-Dx-Clean"
PET_DIR = "/home/data/Lung-PET-CT-Dx_PET"
DICOM_DIR = "/home/data/Lung-PET-CT-Dx_dicom"

def main():
    os.makedirs(PET_DIR, exist_ok=True)

    if not os.path.exists(CLEAN_DIR):
        print(f"Clean directory {CLEAN_DIR} does not exist.")
        return

    patient_folders = [f for f in os.listdir(CLEAN_DIR) if os.path.isdir(os.path.join(CLEAN_DIR, f))]
    print(f"Scanning {len(patient_folders)} patients in {CLEAN_DIR} for PET images...")

    pet_found_count = 0
    moved_count = 0
    deleted_count = 0

    for patient_id in tqdm(patient_folders, desc="Scanning patients"):
        pat_path = os.path.join(CLEAN_DIR, patient_id)
        
        # Images are saved as {series_uid}_image.nii.gz
        image_files = glob.glob(os.path.join(pat_path, "*_image.nii*"))
        
        for img_file in image_files:
            filename = os.path.basename(img_file)
            
            # Extract the raw series_uid
            series_uid = filename.replace("_image.nii.gz", "").replace("_image.nii", "")
            
            # 1. Determine if this image is a PET scan by looking at its original DICOM Modality
            dicom_series_dir = os.path.join(DICOM_DIR, series_uid)
            is_pet = False
            
            if os.path.isdir(dicom_series_dir):
                dcm_files = glob.glob(os.path.join(dicom_series_dir, "*.dcm"))
                if dcm_files:
                    try:
                        # Only need to read header
                        ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
                        if hasattr(ds, 'Modality') and ds.Modality == 'PT':
                            is_pet = True
                    except Exception as e:
                        print(f"Error reading DICOM for {series_uid}: {e}")
            else:
                # Fallback: if the DICOM is missing, check if it already exists in the PET folder
                pet_target_path = os.path.join(PET_DIR, f"{patient_id}_{series_uid}.nii.gz")
                if os.path.exists(pet_target_path):
                    is_pet = True
            
            # 2. If it is a PET scan, handle moving/deleting
            if is_pet:
                pet_found_count += 1
                pet_target_path = os.path.join(PET_DIR, f"{patient_id}_{series_uid}.nii.gz")
                
                # Handle the image
                if os.path.exists(pet_target_path):
                    # Already exists in PET dir -> safely delete from Clean
                    os.remove(img_file)
                    deleted_count += 1
                else:
                    # Not in PET dir yet -> move to PET dir and rename to match existing PET format
                    shutil.move(img_file, pet_target_path)
                    moved_count += 1
                
                # Handle the corresponding mask if it exists
                mask_file = os.path.join(pat_path, f"{series_uid}_mask.nii.gz")
                if not os.path.exists(mask_file):
                    mask_file = os.path.join(pat_path, f"{series_uid}_mask.nii")
                    
                if os.path.exists(mask_file):
                    os.remove(mask_file)

    print("\n--- Scan Complete ---")
    print(f"Total PET images found in Clean dir: {pet_found_count}")
    print(f"Moved to PET folder: {moved_count}")
    print(f"Deleted (already existed in PET folder): {deleted_count}")

if __name__ == "__main__":
    main()
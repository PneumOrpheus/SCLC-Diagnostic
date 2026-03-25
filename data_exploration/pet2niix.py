import os
import glob
import shutil
import pydicom
import subprocess
import tempfile

dicom_root = "/home/data/Lung-PET-CT-Dx_dicom"
nifti_old_root = "/home/data/Lung-PET-CT-Dx"
pet_out_root = "/home/data/Lung-PET-CT-Dx_PET"

os.makedirs(pet_out_root, exist_ok=True)

print(f"Scanning for PET series in {dicom_root}...")

dicom_series_dirs = [os.path.join(dicom_root, d) for d in os.listdir(dicom_root) if os.path.isdir(os.path.join(dicom_root, d))]

pet_series = []

# 1. Identify all PET series
for series_dir in dicom_series_dirs:
    dicom_files = glob.glob(os.path.join(series_dir, "*.dcm"))
    if not dicom_files:
        continue
    
    try:
        ds = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
        if hasattr(ds, 'Modality') and ds.Modality == 'PT':
            series_uid = os.path.basename(series_dir)
            patient_id = str(ds.PatientID).replace("Lung_Dx-", "")
            full_patient_id = str(ds.PatientID)
            pet_series.append({
                'series_dir': series_dir,
                'series_uid': series_uid,
                'patient_id': full_patient_id
            })
    except Exception as e:
        print(f"Error reading {series_dir}: {e}")

print(f"Found {len(pet_series)} PET series to reconvert using dcm2niix4pet.")

# 2 & 3. Process each PET series
for i, item in enumerate(pet_series):
    series_dir = item['series_dir']
    series_uid = item['series_uid']
    patient_id = item['patient_id']
    
    expected_nifti_path = os.path.join(pet_out_root, f"{patient_id}_{series_uid}.nii.gz")
    if os.path.exists(expected_nifti_path):
        print(f"\n[{i+1}/{len(pet_series)}] Skipping {patient_id} - {series_uid}: already exists.")
        continue
    
    print(f"\n[{i+1}/{len(pet_series)}] Processing PET Series: {patient_id} - {series_uid}")
    
    # Convert using dcm2niix4pet to a temporary directory to control the output names safely
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Provide the inner dcm2niix arguments explicitly grouped in a single string under --dcm2niix-options
        cmd = [
            "dcm2niix4pet", 
            series_dir,
            "--destination-path", tmp_dir,
            "--dcm2niix-options", "-z y"
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Find the converted files in tmpdir
            converted_niftis = glob.glob(os.path.join(tmp_dir, "*.nii*"))
            converted_jsons = glob.glob(os.path.join(tmp_dir, "*.json"))
            converted_tsvs = glob.glob(os.path.join(tmp_dir, "*.tsv")) # in case bids creates a tsv
            
            if not converted_niftis:
                print(f"  -> WARNING: No NIfTI generated for {series_uid}")
                continue
                
            # SUCCESS! The new files exist. Now we can safely delete the old corrupted files from the CT folder.
            existing_files_pattern = os.path.join(nifti_old_root, f"{patient_id}_{series_uid}*.*")
            existing_files = glob.glob(existing_files_pattern)
            
            for old_file in existing_files:
                print(f"  -> Successfully generated new PET. Deleting old bad file: {os.path.basename(old_file)}")
                os.remove(old_file)
                
            # Rename and move the new correctly formatted files to designated pet_out_root
            for nifti_file in converted_niftis:
                new_nifti_name = f"{patient_id}_{series_uid}.nii.gz"
                new_nifti_path = os.path.join(pet_out_root, new_nifti_name)
                shutil.move(nifti_file, new_nifti_path)
                print(f"  -> Saved NIfTI to PET folder: {new_nifti_name}")
                
            for json_file in converted_jsons:
                new_json_name = f"{patient_id}_{series_uid}.json"
                new_json_path = os.path.join(pet_out_root, new_json_name)
                shutil.move(json_file, new_json_path)
                print(f"  -> Saved JSON to PET folder: {new_json_name}")
                
            for tsv_file in converted_tsvs:
                new_tsv_name = f"{patient_id}_{series_uid}.tsv"
                new_tsv_path = os.path.join(pet_out_root, new_tsv_name)
                shutil.move(tsv_file, new_tsv_path)
                print(f"  -> Saved TSV to PET folder: {new_tsv_name}")
                
        except subprocess.CalledProcessError as e:
             print(f"  -> ERROR: dcm2niix4pet failed on {series_uid}: code {e.returncode}. Old files were kept intact.")
        except Exception as e:
             print(f"  -> Unexpected ERROR: {e}")

print("\nPET conversion complete!")
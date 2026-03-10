import os
download_path = "/home/data/Lung-PET-CT-Dx/"
print(f"Checking folders in directory: {download_path}")
annotation_path = "/home/data/Annotation/"

missing_anno = []
with os.scandir(download_path) as entries:
    for entry in entries:
        if entry.is_file():
            if entry.name.endswith('.nii.gz'):
                anno = entry.name.split('-')[1].split('_')[0]
                for anno_entry in os.scandir(annotation_path):
                    missing_ann = True
                    if anno_entry.is_dir() and anno_entry.name.startswith(anno):
                        print(f"    -> Found matching annotation file: {anno_entry.name}")
                        missing_ann = False
                        break
                if missing_ann:
                    print(f"    -> Missing annotation for: {entry.name} (Expected annotation ID: {anno})")
                    missing_anno.append(anno)

print(f"Total missing annotations: {len(missing_anno)}, Missing annotation IDs: {missing_anno}")

print("Folder check complete.")

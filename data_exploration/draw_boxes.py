import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib

# Add the root directory to Python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_loader import get_biglunge_data_list, get_lung_pet_ct_dx_data_list 
from data.transforms import get_val_transforms_3d, get_train_transforms_3d

def save_transformed_volumes():
    output_dir = "/home/hansstem/SCLC-Classification/data_exploration/box_images"
    os.makedirs(output_dir, exist_ok=True)

    data_path = "/home/data/BigLunge/pre_formatting_ws_iso1.0mm_croplungs_bb/1"
    
    
    print("Loading data lists for volume generation...")
    splits = get_biglunge_data_list(
        data_path=data_path,
        csv_path="/home/data/BigLunge/patients_parameters.csv",
    )
    for split in ("train", "val", "test"):
        data_list = splits[split]
    
    
    transforms = get_train_transforms_3d(img_size=224, depth_size=128)
    
    saved_count = 0
    for i, sample in enumerate(data_list):
        if saved_count >= 50:
            break
             
        try:
            # Apply transforms
            transformed = transforms(sample)
        except Exception as e:
            print(f"Skipping sample {i} due to transform error: {e}")
            continue
            
        image = transformed["image"]
        image_np = image[0].numpy()  # Get single channel volume
        
        patient_id = sample["patient_id"]
        save_path = os.path.join(output_dir, f"transformed_{patient_id}_d128.nii.gz")
        
        # Save as NIfTI volume
        nii_img = nib.Nifti1Image(image_np, affine=np.eye(4))
        # nib.save(nii_img, save_path)
        
        print(f"Saved volume {save_path}")
        saved_count += 1
        
    print(f"Finished generating {saved_count} volumes in {output_dir}")

def main():
    output_dir = "/home/hansstem/SCLC-Classification/data_exploration/box_images"
    os.makedirs(output_dir, exist_ok=True)

    data_path = "/home/data/Lung-PET-CT-Dx"
    annotation_dir = "/home/data/Annotation_ZMapped_fallback_check"
    
    print("Loading data lists...")
    # Get data lists, test_frac/val_frac don't matter much as long as we get some samples.
    # Setting testing=True artificially limits it for faster loading.
    splits = get_lung_pet_ct_dx_data_list(
        data_path=data_path,
        annotation_dir=annotation_dir,
        testing=False,
        img_size=224
    )
    
    # Gather annotated samples from training set
    annotated_samples = [d for d in splits['train'] if d.get("boxes") is not None and len(d["boxes"]) > 0]
    
    print(f"Found {len(annotated_samples)} annotated samples. Processing the first 10...")
    
    transforms = get_val_transforms_3d(img_size=224, depth_size=64)
    
    saved_count = 0
    for i, sample in enumerate(annotated_samples):
        if saved_count >= 5:
            break
            
        try:
            # Apply transforms
            transformed = transforms(sample)
        except Exception as e:
            print(f"Skipping sample {i} due to transform error: {e}")
            continue
            
        image = transformed["image"]
        mask = transformed["mask"]
        
        # They should be tensors of shape (C, X, Y, Z), taking the 0th channel
        image_np = image[0].numpy()
        mask_np = mask[0].numpy()
        
        # Find the Z slice with the most pixels in the mask (the largest cut of the tumor)
        z_sums = mask_np.sum(axis=(0, 1))
        best_z = z_sums.argmax()
        
        if z_sums[best_z] == 0:
            print(f"Sample {i} (Patient: {os.path.basename(sample['image'])}): Mask is completely empty after extraction.")
            continue
            
        img_slice = image_np[:, :, best_z]
        mask_slice = mask_np[:, :, best_z]
        
        # Display: X is dim 0, Y is dim 1. Transpose for standard PyPlot rendering
        img_slice = img_slice.T
        mask_slice = mask_slice.T
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_slice, cmap='gray')
        axes[0].set_title(f"CT Slice (Z={best_z})")
        axes[0].axis('off')
        
        axes[1].imshow(mask_slice, cmap='gray')
        axes[1].set_title("Generated Mask from Boxes")
        axes[1].axis('off')
        
        axes[2].imshow(img_slice, cmap='gray')
        axes[2].imshow(np.ma.masked_where(mask_slice == 0, mask_slice), cmap='autumn', alpha=0.5)
        axes[2].set_title("Mask Overlay")
        axes[2].axis('off')
        
        patient_name = os.path.basename(sample["image"]).split(".")[0][:15]
        save_path = os.path.join(output_dir, f"box_check_{patient_name}_z{best_z}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved {save_path}")
        saved_count += 1
        
    print(f"Finished generating {saved_count} images in {output_dir}")

if __name__ == "__main__":
    # main()
    print("Starting volume generation...")
    save_transformed_volumes()

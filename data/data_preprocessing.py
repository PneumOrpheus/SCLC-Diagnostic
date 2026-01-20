import os
import numpy as np
import pydicom
import scipy.ndimage
import torch
from typing import List, Tuple, Dict, Optional

def load_dicom_series(dicom_dir: str) -> List[Tuple[np.ndarray, Dict]]:
    """Load a DICOM series from a directory and return the image volume and metadata.

    Loads a full DICOM series from a directory, sorting slices by spatial position.
    
    Args:
        directory (str): Path to folder containing DICOM files.
    
    Returns:
        List: Sorted list of DICOM slice objects.
    """
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist.")
        
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.dcm')]
    if not files:
        raise ValueError(f"No DICOM files found in {directory}")
        
    slices = [pydicom.dcmread(f) for f in files]
    
    # Sort by ImagePositionPatient Z-axis to ensure correct anatomical order and if ImagePositionPatient is missing, fall back to SliceLocation
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: float(x.SliceLocation))
        
    return slices

def convert_to_hounsfield_units(slices: List[pydicom.dataset.FileDataset]) -> np.ndarray:
    """Convert pixel values to Hounsfield Units (HU). Handles the 'padding' values used by 
    scanners for pixels outside the circular field of view.

    Args:
        slices (List[pydicom.dataset.FileDataset]): List of DICOM slice objects.
        
    Returns:
        np.ndarray: 3D numpy array of image data in Hounsfield Units.
    """
    
    # Convert each 2D slice to HU individually, then stack into a 3D volume
    hu_slices = []
    for s in slices:
        slice_image = s.pixel_array.astype(np.int16)

        # Handle rescale parameters on a per-slice basis
        intercept = getattr(s, "RescaleIntercept", 0)
        slope = getattr(s, "RescaleSlope", 1)

        if slope != 1:
            slice_image = slope * slice_image.astype(np.float64)
            slice_image = slice_image.astype(np.int16)

        slice_image += np.int16(intercept)
        hu_slices.append(slice_image)

    image = np.stack(hu_slices).astype(np.int16)
    # Clip typical scanner bounds to clean up artifacts, as per standard 12-bit CT depth 
    image[image < -1024] = -1024
    image[image > 3071] = 3071
    
    return np.array(image, dtype=np.int16)

def resample_volume(image: np.ndarray, scan: List, new_spacing: List[float] = [1.0, 1.0, 1.0]) -> np.ndarray:
    """
    Resamples the 3D volume to an isotropic spacing of 1x1x1 mm.
    This ensures that 3D features are spatially consistent across different patients/scanners.
    
    Args:
        image (np.ndarray): The 3D volume in HU.
        scan (List): The list of DICOM metadata objects.
        new_spacing (List): Target spacing [z, y, x] in mm.
        
    Returns:
        np.ndarray: Resampled 3D volume.
    """
    if not scan:
        raise ValueError("resample_volume received an empty 'scan' list; cannot determine slice thickness.")

    if len(scan) >= 2:
        try:
            slice_thickness = np.abs(
                scan[0].ImagePositionPatient[2] - scan[1].ImagePositionPatient[2]
            )
        except AttributeError:
            slice_thickness = scan[0].SliceThickness
    else:
        # Single-slice scan: fall back to SliceThickness if available
        try:
            slice_thickness = scan[0].SliceThickness
        except AttributeError:
            raise ValueError(
                "Unable to determine slice thickness for single-slice scan: "
                "SliceThickness attribute is missing."
            )
        
    current_spacing = np.array([slice_thickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)
    
    resize_factor = current_spacing / np.array(new_spacing, dtype=np.float32)
    new_real_shape = np.round(image.shape * resize_factor)
    new_shape = new_real_shape.astype(np.int32)
    real_resize_factor = new_shape / image.shape
    
    # Linear interpolation is used for speed and safety against ringing artifacts in high-contrast CTs
    image_resampled = scipy.ndimage.zoom(image, real_resize_factor, order=1)
    
    return image_resampled

def apply_windowing(image: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """
    Apply windowing to the image to enhance contrast for specific tissues.
    Formula: output = (input - (center - width/2)) / width.
    
    Args:
        image (np.ndarray): The 3D volume in HU.
        window_center (float): Center of the window.
        window_width (float): Width of the window.
        
    Returns:
        np.ndarray: Windowed image.
    """
    img_min = window_center - (window_width / 2)
    img_max = window_center + (window_width / 2)
    
    windowed_image = np.clip(image, img_min, img_max)
    
    # Min-Max normalization
    windowed_image = (windowed_image - img_min) / (img_max - img_min)
    
    return windowed_image

def preprocess_sample(dicom_dir: str, output_file: str) -> None:
    """
    Main pipeline function for a single patient scan, which generates a multi-channel 2.5D tensor representation.

    Args:
        dicom_dir (str): Path to the directory containing the DICOM series for a single patient.
        output_file (str): Path to the output file where the processed multi-channel tensor will be saved
            as a compressed NumPy array (e.g., ``.npy``).

    Returns:
        None: The function saves the processed tensor to ``output_file`` and prints a summary message.
    """
    # Load and convert
    slices = load_dicom_series(dicom_dir)
    volume_hu = convert_to_hounsfield_units(slices)
    
    # Resample to Isotropic 1mm
    volume_resampled = resample_volume(volume_hu, slices, new_spacing=[1.0, 1.0, 1.0])
    
    # Multi-Channel Windowing (Best for nodules/parenchyma)
    lung_channel = apply_windowing(volume_resampled, window_center=-600, window_width=1500)
    
    # Mediastinal Window (W:350, L:50) (Best for Lymph Nodes/Soft Tissue)
    mediastinal_channel = apply_windowing(volume_resampled, window_center=50, window_width=350)
    
    # Bone/Wide Window (W:2000, L:300) (Context for chest wall/spine)
    context_channel = apply_windowing(volume_resampled, window_center=300, window_width=2000)
    
    # Stack Channels, allowing direct input to RadImageNet-pretrained backbones
    final_tensor = np.stack([lung_channel, mediastinal_channel, context_channel], axis=-1)
    
    # Save as compressed NumPy array for efficient dataloading
    np.save(output_file, final_tensor)
    print(f"Processed {dicom_dir} -> {output_file} with shape {final_tensor.shape}")
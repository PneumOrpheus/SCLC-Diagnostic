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
    
    # Stack the 2D slices into a 3D volume
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Handle padding values
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope
    
    if slope!= 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
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
    try:
        slice_thickness = np.abs(scan[0].ImagePositionPatient[2] - scan[1].ImagePositionPatient[2])
    except AttributeError:
        slice_thickness = scan[0].SliceThickness
        
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
        window_center (int): Center of the window.
        window_width (int): Width of the window.
        
    Returns:
        np.ndarray: Windowed image.
    """
    img_min = window_center - (window_width // 2)
    img_max = window_center + (window_width // 2)
    
    windowed_image = np.clip(image, img_min, img_max)
    
    # Min-Max normalization
    windowed_image = (windowed_image - img_min) / (img_max - img_min)
    
    return windowed_image
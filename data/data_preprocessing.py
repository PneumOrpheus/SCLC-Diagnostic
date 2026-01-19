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
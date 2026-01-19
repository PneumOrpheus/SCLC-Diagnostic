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
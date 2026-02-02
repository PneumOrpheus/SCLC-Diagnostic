import os
import tempfile
import numpy as np
import nibabel as nib
import scipy.ndimage
import torch
import torch.nn.functional as F
import dicom2nifti
import dicom2nifti.settings as dicom2nifti_settings
import pydicom
from typing import List, Tuple, Optional, Union


"""
SCLC Data Preprocessing Module
------------------------------
Provides preprocessing utilities for CT scan data in NIfTI format (.nii, .nii.gz)
and DICOM format (.dcm or DICOM directories).
Supports multi-channel windowing for RadImageNet-pretrained backbones.
"""


def load_nifti_volume(file_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a NIfTI file and return the image volume and affine matrix.

    Args:
        file_path (str): Path to the NIfTI file (.nii or .nii.gz).

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - 3D numpy array of image data
            - Affine transformation matrix (or None if unavailable)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    if not (file_path.endswith('.nii') or file_path.endswith('.nii.gz')):
        raise ValueError(f"File {file_path} is not a valid NIfTI file.")
    
    nii_img = nib.load(file_path) # type: ignore
    volume = nii_img.get_fdata(dtype=np.float32) # type: ignore
    affine = nii_img.affine # type: ignore
    
    return volume, affine


def load_numpy_volume(file_path: str) -> np.ndarray:
    """Load a numpy file (.npy or .npz) and return the image volume.

    Args:
        file_path (str): Path to the numpy file.

    Returns:
        np.ndarray: Image data array.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    data = np.load(file_path, allow_pickle=True)
    
    # Handle .npz files or dictionary-style .npy files
    if hasattr(data, "item"):
        data_dict = data.item()
        if isinstance(data_dict, dict) and 'scan' in data_dict:
            return data_dict['scan'].astype(np.float32)
        return np.array(data_dict, dtype=np.float32)
    
    return np.array(data, dtype=np.float32)


def is_dicom_file(file_path: str) -> bool:
    """Check if a file is a valid DICOM file.

    Args:
        file_path (str): Path to the file to check.

    Returns:
        bool: True if the file is a valid DICOM file, False otherwise.
    """
    if not os.path.isfile(file_path):
        return False
    
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except Exception:
        return False


def is_dicom_directory(dir_path: str) -> bool:
    """Check if a directory contains DICOM files.

    Args:
        dir_path (str): Path to the directory to check.

    Returns:
        bool: True if the directory contains at least one DICOM file.
    """
    if not os.path.isdir(dir_path):
        return False
    
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if is_dicom_file(file_path):
            return True
    return False


def convert_dicom_to_nifti(dicom_path: str, 
                            output_path: Optional[str] = None,
                            reorient: bool = True) -> str:
    """Convert a DICOM directory or file to NIfTI format.

    Uses dicom2nifti package to perform the conversion.

    Args:
        dicom_path (str): Path to DICOM directory or single DICOM file.
        output_path (str, optional): Path for output NIfTI file. 
            If None, creates a temporary file.
        reorient (bool): Whether to reorient the volume to standard orientation.

    Returns:
        str: Path to the created NIfTI file.

    Raises:
        ValueError: If dicom_path is not a valid DICOM source.
    """
    # Configure dicom2nifti settings
    dicom2nifti_settings.disable_validate_slice_increment()
    dicom2nifti_settings.disable_validate_orthogonal()
    if not reorient:
        dicom2nifti_settings.disable_resampling()
    
    # Determine output path
    if output_path is None:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "converted.nii.gz")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if os.path.isdir(dicom_path):
        # Convert DICOM directory
        if not is_dicom_directory(dicom_path):
            raise ValueError(f"Directory {dicom_path} does not contain valid DICOM files.")
        dicom2nifti.dicom_series_to_nifti(dicom_path, output_path, reorient_nifti=reorient)
    elif os.path.isfile(dicom_path):
        # For single DICOM file, we need to find all files in the same directory
        # that belong to the same series
        if not is_dicom_file(dicom_path):
            raise ValueError(f"File {dicom_path} is not a valid DICOM file.")
        
        parent_dir = os.path.dirname(dicom_path)
        if parent_dir and is_dicom_directory(parent_dir):
            # Convert the entire series from the parent directory
            dicom2nifti.dicom_series_to_nifti(parent_dir, output_path, reorient_nifti=reorient)
        else:
            raise ValueError(
                f"Cannot convert single DICOM file {dicom_path}. "
                "Please provide a directory containing the full DICOM series."
            )
    else:
        raise ValueError(f"Path {dicom_path} does not exist.")
    
    return output_path


def load_dicom_volume(dicom_path: str, 
                       convert_to_nifti_path: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a DICOM directory or file and return the image volume.

    Converts DICOM to NIfTI internally and loads the result.

    Args:
        dicom_path (str): Path to DICOM directory or single DICOM file.
        convert_to_nifti_path (str, optional): Path to save the converted NIfTI file.
            If None, uses a temporary file that is cleaned up after loading.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: 
            - 3D numpy array of image data
            - Affine transformation matrix (or None if unavailable)
    """
    # Convert DICOM to NIfTI
    use_temp = convert_to_nifti_path is None
    nifti_path = convert_dicom_to_nifti(dicom_path, convert_to_nifti_path)
    
    try:
        # Load the converted NIfTI file
        volume, affine = load_nifti_volume(nifti_path)
        return volume, affine
    finally:
        # Clean up temporary file if we created one
        if use_temp and os.path.exists(nifti_path):
            os.remove(nifti_path)
            temp_dir = os.path.dirname(nifti_path)
            if os.path.isdir(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)


def load_volume(file_path: str) -> np.ndarray:
    """Load a volume from NIfTI, numpy, or DICOM format.

    Args:
        file_path (str): Path to the data file or DICOM directory.

    Returns:
        np.ndarray: Image volume as float32 array.
    """
    if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
        volume, _ = load_nifti_volume(file_path)
        return volume
    elif file_path.endswith('.npy') or file_path.endswith('.npz'):
        return load_numpy_volume(file_path)
    elif os.path.isdir(file_path) and is_dicom_directory(file_path):
        volume, _ = load_dicom_volume(file_path)
        return volume
    elif file_path.endswith('.dcm') or is_dicom_file(file_path):
        volume, _ = load_dicom_volume(file_path)
        return volume
    else:
        raise ValueError(f"Unsupported file format: {file_path}. "
                        "Supported formats: .nii, .nii.gz, .npy, .npz, .dcm, or DICOM directory")


def clip_hounsfield_units(volume: np.ndarray, 
                          min_hu: float = -1024, 
                          max_hu: float = 3071) -> np.ndarray:
    """Clip volume to valid Hounsfield Unit range for CT scans.
    
    Standard 12-bit CT depth ranges from -1024 to 3071 HU.

    Args:
        volume (np.ndarray): The 3D volume (assumed to be in HU or raw intensity).
        min_hu (float): Minimum HU value to clip to.
        max_hu (float): Maximum HU value to clip to.

    Returns:
        np.ndarray: Clipped volume.
    """
    return np.clip(volume, min_hu, max_hu)


def resample_volume_isotropic(volume: np.ndarray, 
                               current_spacing: Tuple[float, float, float],
                               target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                               order: int = 1) -> np.ndarray:
    """Resample a 3D volume to isotropic spacing.

    Args:
        volume (np.ndarray): The 3D volume to resample.
        current_spacing (Tuple[float, float, float]): Current voxel spacing (z, y, x) in mm.
        target_spacing (Tuple[float, float, float]): Target voxel spacing in mm.
        order (int): Interpolation order (0=nearest, 1=linear, 3=cubic).

    Returns:
        np.ndarray: Resampled volume.
    """
    current_spacing_array = np.array(current_spacing, dtype=np.float32)
    target_spacing_array = np.array(target_spacing, dtype=np.float32)
    
    resize_factor = current_spacing_array / target_spacing_array
    new_shape = np.round(volume.shape * resize_factor).astype(np.int32)
    real_resize_factor = new_shape / np.array(volume.shape)
    
    return scipy.ndimage.zoom(volume, real_resize_factor, order=order)


def apply_windowing(volume: np.ndarray, 
                    window_center: float, 
                    window_width: float) -> np.ndarray:
    """Apply CT windowing to enhance contrast for specific tissues.

    Windowing formula maps HU values to [0, 1] range based on window center and width.

    Args:
        volume (np.ndarray): The volume in Hounsfield Units.
        window_center (float): Center of the window (level).
        window_width (float): Width of the window.

    Returns:
        np.ndarray: Windowed and normalized image in [0, 1] range.
    """
    img_min = window_center - (window_width / 2)
    img_max = window_center + (window_width / 2)
    
    windowed = np.clip(volume, img_min, img_max)
    windowed = (windowed - img_min) / (img_max - img_min)
    
    return windowed.astype(np.float32)


def create_multichannel_ct(volume: np.ndarray) -> np.ndarray:
    """Create a 3-channel representation using different CT windows.

    Creates three channels optimized for different anatomical structures:
    - Lung window (L:-600, W:1500): Nodules and parenchyma
    - Mediastinal window (L:50, W:350): Lymph nodes and soft tissue
    - Bone/Wide window (L:300, W:2000): Chest wall and spine context

    Args:
        volume (np.ndarray): 3D volume in Hounsfield Units.

    Returns:
        np.ndarray: Multi-channel volume with shape (..., 3) in channel-last format.
    """
    lung_channel = apply_windowing(volume, window_center=-600, window_width=1500)
    mediastinal_channel = apply_windowing(volume, window_center=50, window_width=350)
    bone_channel = apply_windowing(volume, window_center=300, window_width=2000)
    
    return np.stack([lung_channel, mediastinal_channel, bone_channel], axis=-1)


def normalize_intensity(volume: np.ndarray) -> np.ndarray:
    """Normalize volume intensity to [0, 1] range using min-max normalization.

    Handles edge cases where volume has uniform intensity.

    Args:
        volume (np.ndarray): Input volume.

    Returns:
        np.ndarray: Normalized volume in [0, 1] range.
    """
    vol_min = volume.min()
    vol_max = volume.max()
    
    if vol_max > vol_min:
        return (volume - vol_min) / (vol_max - vol_min)
    else:
        # Uniform volume - return zeros to avoid division issues
        return np.zeros_like(volume, dtype=np.float32)


def extract_2d_slice(volume: np.ndarray, 
                     slice_index: Optional[int] = None,
                     axis: int = 0) -> np.ndarray:
    """Extract a 2D slice from a 3D volume.

    Args:
        volume (np.ndarray): 3D volume with shape (D, H, W) or (D, H, W, C).
        slice_index (int, optional): Index of slice to extract. Defaults to middle slice.
        axis (int): Axis along which to slice (0=axial, 1=coronal, 2=sagittal).

    Returns:
        np.ndarray: 2D slice with shape (H, W) or (H, W, C).
    """
    index: int = slice_index if slice_index is not None else volume.shape[axis] // 2
    
    return np.take(volume, index, axis=axis)


def prepare_tensor_for_model(scan: np.ndarray, 
                              img_size: int = 224,
                              convert_to_rgb: bool = True) -> torch.Tensor:
    """Prepare a scan array as a tensor ready for model input.

    Handles dimensionality, channel conversion, and resizing.

    Args:
        scan (np.ndarray): Input scan array (2D or 3D).
        img_size (int): Target image size (height and width).
        convert_to_rgb (bool): Whether to convert grayscale to 3-channel RGB.

    Returns:
        torch.Tensor: Prepared tensor with shape (C, H, W).
    """
    tensor = torch.tensor(scan, dtype=torch.float32)
    
    # Handle different dimensionalities
    if tensor.ndim == 2:
        # (H, W) -> (1, H, W)
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        # Distinguish 2D images with channels from true 3D volumes
        if tensor.shape[-1] in (1, 3):
            # Channel-last image: (H, W, C) -> (C, H, W)
            tensor = tensor.permute(2, 0, 1)
        elif tensor.shape[0] in (1, 3):
            # Channel-first image: (C, H, W), keep as-is
            pass
        else:
            # 3D volume: select middle slice along the largest axis
            depth_axis = int(np.argmax(tensor.shape))
            mid_slice = tensor.shape[depth_axis] // 2
            tensor = tensor.select(dim=depth_axis, index=mid_slice).unsqueeze(0)
    
    # Normalize if not already in [0, 1]
    if tensor.max() > 1.0:
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        if tensor_max > tensor_min:
            tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        else:
            tensor = torch.zeros_like(tensor)
    
    # Convert grayscale to RGB if needed
    if convert_to_rgb and tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    
    # Resize to model's expected input size
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(img_size, img_size),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)
    
    return tensor


def preprocess_nifti_to_numpy(input_path: str, 
                               output_path: str,
                               use_multichannel: bool = True,
                               extract_slice: bool = False,
                               slice_index: Optional[int] = None) -> None:
    """Preprocess a NIfTI file and save as numpy array.

    Main pipeline function for converting raw NIfTI CT scans to preprocessed
    numpy arrays ready for training.

    Args:
        input_path (str): Path to input NIfTI file.
        output_path (str): Path for output numpy file.
        use_multichannel (bool): Whether to create 3-channel windowed output.
        extract_slice (bool): Whether to extract a single 2D slice.
        slice_idx (int, optional): Slice index to extract (default: middle).

    Returns:
        None: Saves processed array to output_path.
    """
    # Load volume
    volume, _ = load_nifti_volume(input_path)
    
    # Clip to valid HU range
    volume = clip_hounsfield_units(volume)
    
    if use_multichannel:
        # Create multi-channel representation
        processed = create_multichannel_ct(volume)
    else:
        # Simple normalization
        processed = normalize_intensity(volume)
    
    if extract_slice:
        processed = extract_2d_slice(processed, slice_index=slice_index, axis=0)
    
    # Save as numpy array
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.save(output_path, processed.astype(np.float32))
    print(f"Processed {input_path} -> {output_path} with shape {processed.shape}")


def preprocess_dicom_to_numpy(input_path: str,
                               output_path: str,
                               use_multichannel: bool = True,
                               extract_slice: bool = False,
                               slice_index: Optional[int] = None,
                               save_nifti: bool = False,
                               nifti_output_path: Optional[str] = None) -> None:
    """Preprocess a DICOM directory/file and save as numpy array.

    Pipeline function for converting DICOM CT scans to preprocessed
    numpy arrays ready for training.

    Args:
        input_path (str): Path to input DICOM directory or file.
        output_path (str): Path for output numpy file.
        use_multichannel (bool): Whether to create 3-channel windowed output.
        extract_slice (bool): Whether to extract a single 2D slice.
        slice_index (int, optional): Slice index to extract (default: middle).
        save_nifti (bool): Whether to also save the intermediate NIfTI file.
        nifti_output_path (str, optional): Path for NIfTI output if save_nifti is True.

    Returns:
        None: Saves processed array to output_path.
    """
    # Determine NIfTI output path if saving
    if save_nifti:
        if nifti_output_path is None:
            nifti_output_path = output_path.replace('.npy', '.nii.gz')
        volume, _ = load_dicom_volume(input_path, convert_to_nifti_path=nifti_output_path)
    else:
        volume, _ = load_dicom_volume(input_path)
    
    # Clip to valid HU range
    volume = clip_hounsfield_units(volume)
    
    if use_multichannel:
        processed = create_multichannel_ct(volume)
    else:
        processed = normalize_intensity(volume)
    
    if extract_slice:
        processed = extract_2d_slice(processed, slice_index=slice_index, axis=0)
    
    # Save as numpy array
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.save(output_path, processed.astype(np.float32))
    print(f"Processed DICOM {input_path} -> {output_path} with shape {processed.shape}")


def batch_convert_dicom_to_nifti(input_dir: str,
                                  output_dir: str,
                                  reorient: bool = True) -> None:
    """Batch convert all DICOM directories to NIfTI files.

    Args:
        input_dir (str): Directory containing DICOM subdirectories.
        output_dir (str): Directory for output NIfTI files.
        reorient (bool): Whether to reorient volumes to standard orientation.

    Returns:
        None: Saves converted NIfTI files to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find DICOM directories
    dicom_dirs = [d for d in os.listdir(input_dir)
                  if os.path.isdir(os.path.join(input_dir, d)) and 
                  is_dicom_directory(os.path.join(input_dir, d))]
    
    if not dicom_dirs:
        print(f"No DICOM directories found in {input_dir}")
        return
    
    for dirname in dicom_dirs:
        dicom_path = os.path.join(input_dir, dirname)
        output_path = os.path.join(output_dir, f"{dirname}.nii.gz")
        
        try:
            convert_dicom_to_nifti(dicom_path, output_path, reorient=reorient)
            print(f"Converted {dicom_path} -> {output_path}")
        except Exception as e:
            print(f"Error converting {dirname}: {e}")


def batch_preprocess_directory(input_dir: str,
                                output_dir: str,
                                use_multichannel: bool = True,
                                extract_slice: bool = False) -> None:
    """Batch preprocess all NIfTI and DICOM files/directories in a directory.

    Args:
        input_dir (str): Directory containing NIfTI files or DICOM subdirectories.
        output_dir (str): Directory for output numpy files.
        use_multichannel (bool): Whether to create 3-channel windowed output.
        extract_slice (bool): Whether to extract single 2D slices.

    Returns:
        None: Saves processed arrays to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find NIfTI files
    nifti_files = [f for f in os.listdir(input_dir) 
                   if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    # Find DICOM directories
    dicom_dirs = [d for d in os.listdir(input_dir)
                  if os.path.isdir(os.path.join(input_dir, d)) and 
                  is_dicom_directory(os.path.join(input_dir, d))]
    
    # Find individual DICOM files
    dicom_files = [f for f in os.listdir(input_dir)
                   if f.endswith('.dcm') or 
                   (os.path.isfile(os.path.join(input_dir, f)) and 
                    is_dicom_file(os.path.join(input_dir, f)))]
    
    if not nifti_files and not dicom_dirs and not dicom_files:
        print(f"No NIfTI or DICOM files found in {input_dir}")
        return
    
    # Process NIfTI files
    for filename in nifti_files:
        input_path = os.path.join(input_dir, filename)
        output_filename = filename.replace('.nii.gz', '.npy').replace('.nii', '.npy')
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            preprocess_nifti_to_numpy(
                input_path, 
                output_path,
                use_multichannel=use_multichannel,
                extract_slice=extract_slice
            )
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Process DICOM directories
    for dirname in dicom_dirs:
        input_path = os.path.join(input_dir, dirname)
        output_filename = f"{dirname}.npy"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            preprocess_dicom_to_numpy(
                input_path,
                output_path,
                use_multichannel=use_multichannel,
                extract_slice=extract_slice
            )
        except Exception as e:
            print(f"Error processing DICOM directory {dirname}: {e}")
    
    # Process individual DICOM files
    processed_dicom_parents = set()
    for filename in dicom_files:
        input_path = os.path.join(input_dir, filename)
        parent_dir = os.path.dirname(input_path)
        
        # Avoid processing the same parent directory multiple times
        if parent_dir in processed_dicom_parents:
            continue
        processed_dicom_parents.add(parent_dir)
        
        output_filename = f"{os.path.basename(parent_dir)}_dicom.npy"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            preprocess_dicom_to_numpy(
                parent_dir,
                output_path,
                use_multichannel=use_multichannel,
                extract_slice=extract_slice
            )
        except Exception as e:
            print(f"Error processing DICOM file {filename}: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess CT scans for SCLC classification")
    parser.add_argument("--input", type=str, required=True, help="Input NIfTI/DICOM file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output numpy/NIfTI file or directory")
    parser.add_argument("--multichannel", action="store_true", default=True,
                        help="Create 3-channel windowed output")
    parser.add_argument("--extract-slice", action="store_true",
                        help="Extract middle 2D slice instead of full volume")
    parser.add_argument("--batch", action="store_true",
                        help="Process entire directory")
    parser.add_argument("--convert-dicom", action="store_true",
                        help="Convert DICOM to NIfTI only (no preprocessing)")
    parser.add_argument("--save-nifti", action="store_true",
                        help="Also save intermediate NIfTI when processing DICOM")
    
    args = parser.parse_args()
    
    if args.convert_dicom:
        # DICOM to NIfTI conversion only
        if args.batch:
            batch_convert_dicom_to_nifti(args.input, args.output)
        else:
            output_path = convert_dicom_to_nifti(args.input, args.output)
            print(f"Converted DICOM to NIfTI: {output_path}")
    elif args.batch:
        batch_preprocess_directory(
            args.input, 
            args.output,
            use_multichannel=args.multichannel,
            extract_slice=args.extract_slice
        )
    else:
        # Determine input type and process accordingly
        if os.path.isdir(args.input) and is_dicom_directory(args.input):
            preprocess_dicom_to_numpy(
                args.input,
                args.output,
                use_multichannel=args.multichannel,
                extract_slice=args.extract_slice,
                save_nifti=args.save_nifti
            )
        elif args.input.endswith('.dcm') or is_dicom_file(args.input):
            preprocess_dicom_to_numpy(
                args.input,
                args.output,
                use_multichannel=args.multichannel,
                extract_slice=args.extract_slice,
                save_nifti=args.save_nifti
            )
        else:
            preprocess_nifti_to_numpy(
                args.input,
                args.output,
                use_multichannel=args.multichannel,
                extract_slice=args.extract_slice
            )

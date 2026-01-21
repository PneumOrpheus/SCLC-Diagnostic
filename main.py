import argparse
import pydicom
import dicom2nifti
import os
import dicom2nifti.settings as settings

dicom_file_path = '../work/BIG_LUNGE/CT_images/1/DICOM/000066EC/AAC37262/AA143FD3/00007DE1/'
nifti_output_path = './NIFTI_data'

try:
    os.makedirs(nifti_output_path, exist_ok=True)

    dicom2nifti.convert_directory(
        dicom_file_path,
        nifti_output_path,
        compression=True,
        reorient=True
    )
except Exception as error:
    print("Error reading DICOM file:", error)


def parse_option():
    args = argparse.ArgumentParser()

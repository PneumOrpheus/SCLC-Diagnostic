import pydicom

dicom_file_path = '../work/BIG_LUNGE/CT_images/1/DICOM/000066EC/AAC37262/AA143FD3/00007DE1/EE000524'
try:
    ds = pydicom.filereader.dcmread(dicom_file_path)
    print(ds)
except Exception as error:
    print("Error reading DICOM file:", error)

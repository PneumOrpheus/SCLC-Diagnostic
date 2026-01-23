import argparse
# import pydicom
#import dicom2nifti
import os
# import dicom2nifti.settings as settings
from models.config import get_config
from models.utils import load_pretrained
from models.build import build_model

dicom_file_path = '../work/BIG_LUNGE/CT_images/1/DICOM/000066EC/AAC37262/AA143FD3/00007DE1/'
nifti_output_path = './NIFTI_data'

# try:
#     os.makedirs(nifti_output_path, exist_ok=True)

#     dicom2nifti.convert_directory(
#         dicom_file_path,
#         nifti_output_path,
#         compression=True,
#         reorient=True
#     )
# except Exception as error:
#     print("Error reading DICOM file:", error)


def parse_option():
    args = argparse.ArgumentParser('Swin Transformer training and evaluation script')
    args, unparsed = args.parse_known_args()
    config = get_config(args)
    return args, config

def main(config):
    model = build_model(config)
    # model.cuda()
    if config.MODEL.PRETRAINED: 
        load_pretrained(config, model, None)
        print("Nå gooder vi")


    print("Er vi good nå?")

if __name__ == '__main__':
    args, config = parse_option()
    # the config file is in /home/hansstem/work/sclc-pretrained/RadImageNet_swin/rin_config.yaml and we are in /home/hansstem/work/sclc-hansstem/SCLC-Classification
    # get the file 

    config_file_path = '../RadImageNet_swin/rin_config.yaml'
    config.merge_from_file(config_file_path)

    # print("Config:", config)
    # print("Args:", args)
    main(config)

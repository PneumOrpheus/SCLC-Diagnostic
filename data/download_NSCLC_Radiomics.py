import pandas as pd
from tcia_utils import nbia

collections = nbia.getCollections()

for collection in collections:
    for key, value in collection.items():
        if value == "NSCLC-Radiomics":
            print(f"{key}: {value}")
            

data = nbia.getSeries(collection = "NSCLC-Radiomics")


download_path = "/home/data/NSCLC-Radiomics"

nbia.downloadSeries(data, path=download_path)

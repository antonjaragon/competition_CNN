#%%
# Download cats and dogs dataset into 'data' folder

import os
import kaggle


# Download dataset
kaggle.api.competition_download_files('dogs-vs-cats', quiet=False)

#%%
# Unzip dataset

import zipfile

with zipfile.ZipFile('dogs-vs-cats.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

with zipfile.ZipFile('data/train.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

with zipfile.ZipFile('data/test1.zip', 'r') as zip_ref:
    zip_ref.extractall('data')

# %%

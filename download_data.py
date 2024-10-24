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
    os.rename('data/train', 'data/full_dataset')


# %%

# split train data into train and test sets
import os
import glob
import shutil
import random
import tqdm


# remove previous data folders and children
if os.path.exists('data/train'):
    shutil.rmtree('data/train')
if os.path.exists('data/test'):
    shutil.rmtree('data/test')


# Create directories
os.makedirs('data/train/cats')
os.makedirs('data/train/dogs')
os.makedirs('data/test/cats')
os.makedirs('data/test/dogs')


seed = 42
random.seed(seed)

# Get all files
cat_files = glob.glob('data/full_dataset/cat.*.jpg')
dog_files = glob.glob('data/full_dataset/dog.*.jpg')

# Shuffle files
random.shuffle(cat_files)
random.shuffle(dog_files)

# Split files

test_samples = 1000
train_cat_files = cat_files[test_samples:]
test_cat_files = cat_files[:test_samples]
train_dog_files = dog_files[test_samples:]
test_dog_files = dog_files[:test_samples]


# copy them to the corresponding folders
for file in tqdm.tqdm(train_cat_files):
    shutil.copy(file, 'data/train/cats')

for file in tqdm.tqdm(test_cat_files):
    shutil.copy(file, 'data/test/cats')

for file in tqdm.tqdm(train_dog_files):
    shutil.copy(file, 'data/train/dogs')

for file in tqdm.tqdm(test_dog_files):
    shutil.copy(file, 'data/test/dogs')



# %%
print(f"Number of training cat images: {len(train_cat_files)}")
print(f"Number of test cat images: {len(test_cat_files)}")
print(f"Number of training dog images: {len(train_dog_files)}")
print(f"Number of test dog images: {len(test_dog_files)}")

# %%
import pathlib
import os
import numpy as np
import pandas as pd
import random

def process_path(file_path):
    img = np.load(file_path)
#     img = img / (img.max(axis=(0,1)) + 1e-6) ## pre-processing has been done in trainer.py
    label = img
    lst = file_path.split('/')
    img_name = lst[-2]
    single_cell_image_name = lst[-1]
    return img, label, single_cell_image_name, img_name


image_dir = image_dir = '/workspace/cp2image/seg_images_by_nuclei_center/'
multi_imagelss = list(filter(lambda x: not x.startswith('.'), os.listdir(image_dir)))
single_imagels = []
for multi_imagels in multi_imagelss:
    sinimage_names = os.listdir(os.path.join(image_dir, multi_imagels))
    for sinimage_name in sinimage_names:
        single_imagels.append(sinimage_name)
random.Random(33).shuffle(single_imagels)
print('-------------Single image names read----------')
print('number of single images read: {}'.format(len(single_imagels)))

train_part = int(len(single_imagels) * 0.8)
val_part = int(len(single_imagels) * 0.1)
test_part = len(single_imagels) - train_part - val_part
train_singlenames = np.array(single_imagels[:train_part])
val_singlenames = np.array(single_imagels[train_part:train_part+val_part])
test_singlenames = np.array(single_imagels[train_part+val_part:])

moa_data = pd.read_csv('/workspace/cp2image/BBBC021_v1_moa.csv')
img_metadata = pd.read_csv('/workspace/cp2image/BBBC021_v1_image.csv')
merged = pd.merge(img_metadata, moa_data,
                  left_on=['Image_Metadata_Compound', 'Image_Metadata_Concentration'], 
                  right_on=['compound', 'concentration'])
merged['treatment'] = merged['compound'] + '_' + merged['concentration'].astype(str)

train_df = pd.DataFrame(train_singlenames, columns=['singlename'])
train_df['multiname'] = train_df['singlename'].str.rsplit('_', 1).str[0] + '.tif'
train_df = pd.merge(train_df, merged[['Image_FileName_DAPI','treatment']], left_on='multiname', right_on='Image_FileName_DAPI')
train_grouped = train_df[['singlename', 'treatment']].groupby('treatment')
print('----------------train group created------------')

val_df = pd.DataFrame(val_singlenames, columns=['singlename'])
val_df['multiname'] = val_df['singlename'].str.rsplit('_', 1).str[0] + '.tif'
val_df = pd.merge(val_df, merged[['Image_FileName_DAPI','treatment']], left_on='multiname', right_on='Image_FileName_DAPI')
val_grouped = val_df[['singlename', 'treatment']].groupby('treatment')
print('----------------val group created------------')

test_df = pd.DataFrame(test_singlenames, columns=['singlename'])
test_df['multiname'] = test_df['singlename'].str.rsplit('_', 1).str[0] + '.tif'
test_df = pd.merge(test_df, merged[['Image_FileName_DAPI','treatment']], left_on='multiname', right_on='Image_FileName_DAPI')
train_size = train_df.shape[0]
val_size = val_df.shape[0]
test_size = test_df.shape[0]
print('>> train size: {}, val size: {}, test size: {}'.format(train_df.shape[0], val_df.shape[0], test_df.shape[0]))

features_all = pd.read_csv('/workspace/cp2image/segbynucleicenter_CP_moa_undemean_standarizedbycompound_features_withxylocation.csv', index_col=0)
features_all = features_all.set_index('single_img_name')

def generateBatch():
    single_name_ls = list(train_grouped.apply(lambda x: x.sample(1))['singlename'])
    single_path_ls = [os.path.join(image_dir, single_name.rsplit('_', 1)[0], single_name) for single_name in single_name_ls]
    image_lst = [process_path(single_path)[0] for single_path in single_path_ls]
    image_batch = np.array(image_lst)
    feature_batch = features_all.loc[single_name_ls].iloc[:,4:-9].to_numpy().reshape([-1,1,1,453])
    return image_batch, feature_batch


train_count = 0
def generateTrainBatch():
    global train_count
    single_name_ls = list(train_df['singlename'][train_count*104:(train_count+1)*104])
    single_path_ls = [os.path.join(image_dir, single_name.rsplit('_', 1)[0], single_name) for single_name in single_name_ls]
    item_lst = [process_path(single_path) for single_path in single_path_ls]
    image_batch = np.array([item[0] for item in item_lst])
    singlename_batch = np.array([item[2] for item in item_lst])
    imagename_batch = np.array([item[3] for item in item_lst])
    train_count += 1
    if train_count == 100:
        print('train_count correct')
    return image_batch, singlename_batch, imagename_batch

val_count = 0
def generateValBatch():
    global val_count
    single_name_ls = list(val_df['singlename'][val_count*104:(val_count+1)*104])
    single_path_ls = [os.path.join(image_dir, single_name.rsplit('_', 1)[0], single_name) for single_name in single_name_ls]
    item_lst = [process_path(single_path) for single_path in single_path_ls]
    image_batch = np.array([item[0] for item in item_lst])
    singlename_batch = np.array([item[2] for item in item_lst])
    imagename_batch = np.array([item[3] for item in item_lst])
    feature_batch = features_all.loc[single_name_ls].iloc[:,4:-9].to_numpy().reshape([-1,1,1,453])
    val_count += 1
    if val_count >= 48800//104:
        val_count = 0
    return image_batch, feature_batch


test_count = 0
def generateTestBatch():
    global test_count
    single_name_ls = list(test_df['singlename'][test_count*104:(test_count+1)*104])
    single_path_ls = [os.path.join(image_dir, single_name.rsplit('_', 1)[0], single_name) for single_name in single_name_ls]
    item_lst = [process_path(single_path) for single_path in single_path_ls]
    image_batch = np.array([item[0] for item in item_lst])
    singlename_batch = np.array([item[2] for item in item_lst])
    imagename_batch = np.array([item[3] for item in item_lst])
    feature_batch = features_all.loc[single_name_ls].iloc[:,4:-9].to_numpy().reshape([-1,1,1,453])
    test_count += 1
    if test_count >= 48800 // 104:
        test_count = 0
    return image_batch, feature_batch, singlename_batch, imagename_batch



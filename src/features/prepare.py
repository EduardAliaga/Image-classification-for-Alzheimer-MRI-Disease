from PIL import Image
import io
import os
import gc
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from tensorflow.keras import layers
from pathlib import Path
import pickle
import sys

"""Define function to prepare the data from the given dataset"""
def prepare_data(dataset_path, data_augmentation=False):
    # read data that comes from a parquet file
    dataset = pd.read_parquet(dataset_path)

    # compute number of instances in the dataset
    num_instances = len(dataset)
    if data_augmentation:
        num_instances *=2

    # initialize some lists to store the data
    dataset2 = [[0,''] for i in range(num_instances)]
    images = ['*' for i in range(num_instances)]
    labels = [-1 for i in range(num_instances)]

    # define image transformation for data preparation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to (224, 224)
        transforms.ToTensor(),          # Convert PIL Image to tensor
    ])

    if data_augmentation:
        for i in range(0,len(dataset)):
            if dataset.iloc[i]['label'] not in [2,3]:
                image_bytes = dataset.iloc[i]['image']['bytes'] # Get bytes
                image_pil = Image.open(io.BytesIO(image_bytes)) # Convert bytes to PIL Image

                # apply random brightness augmentation
                aug_image = tf.image.stateless_random_brightness(image_pil, max_delta=np.random.rand(), seed=(3,0))
                pil_format = Image.fromarray(aug_image.numpy())

                # store transformed data
                dataset2[2*i] = [transform(image_pil),dataset.iloc[i]['label']]
                images[2*1] = transform(image_pil)
                labels[2*i] = dataset.iloc[i]['label']
                dataset2[2*i+1] = [transform(pil_format),dataset.iloc[i]['label']]
                images[2*i+1] = transform(pil_format)
                labels[2*i+1] = dataset.iloc[i]['label']

    else:
        for i in range(0,len(dataset)):
            image_bytes = dataset.iloc[i]['image']['bytes'] # Get bytes
            image_pil = Image.open(io.BytesIO(image_bytes)) # Convert bytes to PIL Image

            # store transformed data
            dataset2[i] = [transform(image_pil),dataset.iloc[i]['label']]
            images[i] = transform(image_pil)
            labels[i] = dataset.iloc[i]['label']

    # returned prepared (preprocessed) images and labels
    return images,labels


"""Define function to prepare training and test data"""
def prepare(test_path, train_path):

    data_augmentation=False

    # prepare the data (training and test)
    train_img,train_lab = prepare_data(train_path+'/train.parquet', data_augmentation)
    test_img,test_lab = prepare_data(test_path+'/test.parquet')

    # Debugging: Print the lengths of the data
    print("Train data length:", len(train_img))
    print("Test data length:", len(test_img))

    # define output paths for prepared data
    prepared_path_train = "data/prepared_data/train"
    prepared_path_test =  "data/prepared_data/test"

    # save train and test prepared data to pickle files
    with open (prepared_path_train+'/train.pkl','wb') as file:
       pickle.dump((train_img,train_lab), file)

    with open (prepared_path_test+'/test.pkl','wb') as file:
       pickle.dump((test_img,test_lab), file)


# Call the prepare function with command line arguments
prepare(sys.argv[1],sys.argv[2])

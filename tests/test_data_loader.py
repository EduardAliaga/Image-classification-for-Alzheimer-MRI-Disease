import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import pickle
import mlflow
import mlflow.pytorch
from PIL import Image
import pandas as pd
import io
import gc
import numpy as np
import pytest

from configuration_test import sample_dataset
from configuration_test import train_data_loader

"""Create fixture with sampled dataset"""
@pytest.fixture
def sample_data():
    return sample_dataset()


"""Check if the sample dataset has images"""
def test_dataset_has_images(sample_data):
    assert len(sample_data.image_tensors) > 0, "Sample data should have images"


"""Check if the sample dataset has labels"""
def test_dataset_has_labels(sample_data):
    assert len(sample_data.labels) > 0, "Sample data should have labels"


"""Check if the images are tensors"""
def test_image_is_valid(sample_data):
    for image in sample_data.image_tensors:
        if torch.is_tensor(image):
            return True
        else:
            raise ValueError("Image is not a tensor")


"""Check if labels are valid (either 0, 1, 2 or 3)"""
def test_label_is_valid(sample_data):
    valid_labels = [0, 1, 2, 3]

    for label in sample_data.labels:
        if label in valid_labels:
            return True
        else:
            raise ValueError("Label not in available categories")


"""Check if the train loader has any rows. """
def test_train_loader_has_rows(sample_data):
    train_loader, valid_loader = train_data_loader(dataset=sample_data, batch_size=64)
    assert len(train_loader) > 0, "Train loader should have rows"


"""Check if the validation loader has any rows. """
def test_val_loader_has_rows(sample_data):
    train_loader, valid_loader = train_data_loader(dataset=sample_data, batch_size=64)
    assert len(valid_loader) > 0, "Validation loader should have rows"

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

from configuration_test import ResidualBlock
from configuration_test import ResNet
from configuration_test import specify_resnet

"""Define fixture with resnet specifications"""
@pytest.fixture
def resnet_specifications():
    #model, params, criterion, optimizer = specify_resnet()
    return specify_resnet()


""""Check if the model parameters are defined correctly: that the number of output
features is the same as number of classes"""
def test_out_features_same_as_num_classes(resnet_specifications):
    model, params, criterion, optimizer = resnet_specifications

    # check it has the same number of output features as number of classes (4)
    num_classes = params['num_classes']
    assert model.fc.out_features == num_classes, "The number of output features should be the same as the number of classes"


"""Check if the model parameters are defined correctly: the batch size is a power of 2"""
def test_model_parameters(resnet_specifications):
    model, params, criterion, optimizer = resnet_specifications

    batch_size = params['batch_size']
    assert (batch_size & (batch_size - 1)) == 0, "The batch size should be a power of 2"

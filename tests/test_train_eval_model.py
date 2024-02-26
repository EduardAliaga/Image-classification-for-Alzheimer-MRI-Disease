import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import pickle
import mlflow
import mlflow.pytorch
from pathlib import Path
from PIL import Image
import pandas as pd
import io
import gc
import numpy as np
import pytest
import json
import dvc.api

from configuration_test import sample_dataset
from configuration_test import train_data_loader
from configuration_test import specify_resnet
from configuration_test import save_model
from configuration_test import do_experiment

"""Declare fixture that returns a vector with the train losses obtained in the
different training epochs and the validation accuracy obtained."""
@pytest.fixture
def losses_and_accuracies():
    sample_data = sample_dataset()
    train_loader, valid_loader = train_data_loader(dataset=sample_data, batch_size=64)
    model, params, criterion, optimizer = specify_resnet()

    total_step = len(train_loader)
    with mlflow.start_run():
        train_losses, validation_accuracy = do_experiment(train_loader,valid_loader,model,params,criterion,optimizer,str(1))

    return train_losses, validation_accuracy


"""Test the model training: if the training loss is not None"""
def test_losses_not_none(losses_and_accuracies):
    train_losses, validation_accuracy = losses_and_accuracies
    assert train_losses is not None, "Train loss should not be None"


"""Test the model validation: if the validation accuracy is not None"""
def test_accuracy_not_none(losses_and_accuracies):
    train_losses, validation_accuracy = losses_and_accuracies
    assert validation_accuracy is not None, "Validation accuracy should not be None"


"""Check that the training model improves: that the training loss does not increase
significantly after some epochs"""
def test_loss_not_increasing_significantly(losses_and_accuracies):
    train_losses, validation_accuracy = losses_and_accuracies

    for i in range(len(train_losses) - 1):
        # only check when the model has had time to learn something
        if i > 6:
            loss_increase = train_losses[i + 1] - train_losses[i]
            # check the training loss does not increase by more than 10% between epochs
            if (loss_increase / train_losses[i - 1]) > 0.1:
                raise ValueError("Training loss increased significantly")
    return True


"""Check that the training model improves: that the last training loss is smaller
than the first one"""
def test_loss_decreased(losses_and_accuracies):
    train_losses, validation_accuracy = losses_and_accuracies

    if train_losses[1] < train_losses[len(train_losses)-1]:
        return ValueError("Training loss should have decreased")
    else:
        return True


"""Check that the accuracy has reasonable values (between 0 and 100%)"""
def test_accuracy_is_reasonable(losses_and_accuracies):
    train_losses, validation_accuracy = losses_and_accuracies

    assert 0.0 <= validation_accuracy <= 100.0, "Validation accuracy should be between 0 and 100%"


"""Test metrics correctly stored: accuracies stored have values between 0 and 100%"""
def test_metrics_stored_correctly():
    ROOT_DIR= Path(Path(__file__).resolve().parent).parent
    METRICS_DIR = ROOT_DIR / "metrics"

    with open(METRICS_DIR / "scoresBest.json", "r") as scores_file:
        loaded_metrics = json.load(scores_file)

    # Check if the values in loaded_metrics are between 0 and 100
    for key, value in loaded_metrics.items():
        assert 0 <= value <= 100, "Accuracy is not between 0 and 100"

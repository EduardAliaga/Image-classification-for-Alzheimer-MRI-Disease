"""This script configures the environment for the different tests to be performed"""

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

"""This script contains different classes and functions as defined in the train
and evaluate scripts of the project. Nevertheless, some of them have some modifications
to facilitate the testings."""


"""Alzheimer_Dataset class that also has a select function that returns only some
rows from the given dataset"""
class AlzheimerDataset(Dataset):
    def __init__(self,image_tensors,labels,transform=None):
        self.image_tensors = image_tensors
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        image = self.image_tensors[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    # add select function to be able to select some of the rows from the given dataset
    def select(self, indices):
        selected_images = [self.image_tensors[i] for i in indices]
        selected_labels = [self.labels[i] for i in indices]
        return AlzheimerDataset(selected_images, selected_labels, self.transform)


"""Returns a smaller dataset for training and testing Alzheimer_MRI dataset"""
def sample_dataset():

    train_pkl_path = '../data/prepared_data/train/train.pkl'
    test_pkl_path = '../data/prepared_data/test/test.pkl'

    # load pkl files
    with open(train_pkl_path,'rb') as tr_file:
        image_tensors_tr,labels_tr = pickle.load(tr_file)

    with open(test_pkl_path,'rb') as test_file:
        image_tensors_test,labels_test = pickle.load(test_file)

    # create dataset objects
    dataset_train = AlzheimerDataset(image_tensors_tr,labels_tr)
    dataset_test = AlzheimerDataset(image_tensors_test,labels_test)

    # create smaller sample data for testing
    sample_data = dataset_train.select(list(range(25)))  # select first 25 samples
    return sample_data


def train_data_loader(dataset,batch_size,random_seed=42,valid_size=0.2,shuffle=True):

    # load the dataset
    train_dataset = dataset
    valid_dataset = dataset

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 4):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


"""Apart from the model and optimizer, it returns a vector with the epoch
training losses """
def train(train_loader, model,criterion,optimizer,params,device):

    epoch_losses = []

    for epoch in range(params['num_epochs']):
        epoch_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

            epoch_loss += loss.item()

        # Compute and store train loss
        mlflow.log_metric('train_loss',loss.item())
        epoch_losses.append(epoch_loss)

    return model,optimizer,epoch_losses


"""Returns the achieved validation accuracy"""
def validation(valid_loader,model,device):

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        accuracy = 100*correct/total
        mlflow.log_metric('val_acc',accuracy)
        return accuracy


def save_model(model,optimizer,name):
    checkpoint = {'model': model,
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict()}
    torch.save(checkpoint, name+'.pth')


def do_experiment(train_loader,valid_loader,model,params,criterion,optimizer,idx):
    mlflow.log_params(params)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,optimizer,epoch_losses = train(train_loader, model,criterion,optimizer,params,device)

    accuracy = validation(valid_loader,model,device)

    mlflow.set_tag("Experiment group","Experimento"+idx)

    save_model(model,optimizer,'Model'+idx)

    return epoch_losses, accuracy


"""Specifies the ResNet: the model, parameters, criterion and optimizer"""
def specify_resnet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)

    params= {
            'num_classes':4,
            'num_epochs':15,
            'batch_size': 32,
            'learning_rate':0.01
        }

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'], weight_decay = 0.001, momentum = 0.9)

    return model, params, criterion, optimizer

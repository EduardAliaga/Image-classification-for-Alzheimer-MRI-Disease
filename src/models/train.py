import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from codecarbon import EmissionsTracker, track_emissions
import dagshub
import mlflow, mlflow.pytorch
from pathlib import Path
import pickle
import gc
import numpy as np
import yaml
import json

# Path of the root
ROOT_DIR= Path(Path(__file__).resolve().parent.parent).parent
# Path to the processed data folder
PROCESSED_DATA_DIR = ROOT_DIR / "data/prepared_data"
# Path to the metrics folder
METRICS_DIR = ROOT_DIR / "metrics"
# Path to the models folder
MODELS_FOLDER_PATH = ROOT_DIR / "models"

"""Establish a connection to DagsHub"""
print("Establish connection")
dagshub.init("taed2-ML-Alphas", "aligator241", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/aligator241/taed2-ML-Alphas.mlflow")


"""Define a dataset class for AlzheimerDataset for a given dataset.
It returns the corresponding image and label"""
class AlzheimerDataset(Dataset):
    def __init__(self,image_tensors,labels,transform=None):
        self.image_tensors = image_tensors
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        image = self.image_tensors[idx]
        label = self.labels[idx]

        return image,label


"""Define a data loader function for training.
Input parameters: dataset, batch_size, random_seed, valid_size and shuffle.
Output: train and validation loaders."""
def train_data_loader(dataset,
                batch_size,
                random_seed=42,
                valid_size=0.2,
                shuffle=True):

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


"""Define a residual block class for the ResNet model"""
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


"""Define the ResNet model"""
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


"""Define a function for training the model with emissions tracking.
Input parameters are: train_loader, model,criterion,optimizer,params and device
It returns the model and optimizer. """
@track_emissions
def train(train_loader, model,criterion,optimizer,params,device):

    print("Start training")

    total = 0
    correct = 0
    for epoch in range(params['num_epochs']):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Compute accuracy
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

        # Compute and store train loss
        accuracy = 100 * correct/total
        mlflow.log_metric('train_accuracy',accuracy)
        mlflow.log_metric('train_loss',loss.item())
        print ('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                       .format(epoch+1, params['num_epochs'], loss.item(), accuracy))

    return model,optimizer,accuracy

"""Define the function to validate the model
Input parameters: validation loader, model and device.
It logs the achieved accuracy."""
def validation(valid_loader,model,device):

    print("Start validation")

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
        
        val_acc = 100*correct/total
        mlflow.log_metric('val_acc',val_acc)
        print('Accuracy of the network on the {} validation images: {} %'.format(5000, val_acc))
    
    return val_acc


"""Define function that saves the model, given the model, optimizer and na,e"""
def save_model(model,optimizer,path):
    print("Saving model")
    checkpoint = {'model': model,
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict()}

    torch.save(checkpoint['state_dict'], path / 'alzheimerModel.zip')


"""Define steps of the experiment"""
def main():

    # Start the emissions tracker
    tracker = EmissionsTracker()
    tracker.start()

    print('Logging')
    mlflow.autolog()
    mlflow.pytorch.autolog()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read train and test data
    print("Step1: Reading .pkl files")

    #train_path_kaggle = '/kaggle/input/images/train.pkl'
    with open(PROCESSED_DATA_DIR / "train/train.pkl",'rb') as tr_file:
        image_tensors_tr,labels_tr = pickle.load(tr_file)

    # Create dataset objects
    print("Step2: Creating Dataset objects")
    dataset_train = AlzheimerDataset(image_tensors_tr,labels_tr)

    # Load parametres
    with open (ROOT_DIR /'params.yaml','r') as file:
        params = yaml.safe_load(file)
        # default params= { 'num_classes':4, 'num_epochs':20, 'batch_size': 64, 'learning_rate':0.01}
    
    print("Using the following parameters")
    print(params)

    # Create loaders
    print("Step3: Creating loaders ")
    trainLoader, validLoader = train_data_loader(dataset_train,batch_size=params['batch_size'],shuffle=True)

    print("Step4: Creating ResNet")
    model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)

    # Initialize loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'], weight_decay = 0.001, momentum = 0.9)

    metrics_dict = {}
    # Train the model
    print("Start run")
    with mlflow.start_run():
        mlflow.log_params(params)
        print("Step5: Start training")
        model,optimizer,train_acc = train(trainLoader, model,criterion,optimizer,params,device)
        val_acc = validation(validLoader,model,device)
        metrics_dict = {"train_accuracy:":train_acc, "validation_accuracy":val_acc}
        save_model(model,optimizer,MODELS_FOLDER_PATH)

    # Stop the emissions tracker
    tracker.stop()

    with open(METRICS_DIR / "scores.json", "w") as scores_file:
        json.dump(
            metrics_dict,
            scores_file,
            indent=4,
        )

main()

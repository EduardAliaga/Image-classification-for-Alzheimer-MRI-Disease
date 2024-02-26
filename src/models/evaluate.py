import json
import pickle
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import zipfile
import io


# Path of the root
ROOT_DIR= Path(Path(__file__).resolve().parent.parent).parent
# Path to the processed data folder
PROCESSED_DATA_DIR = ROOT_DIR / "data/prepared_data"
# Path to the metrics folder
METRICS_DIR = ROOT_DIR / "metrics"
# Path to the models folder
MODELS_FOLDER_PATH = ROOT_DIR / "models"


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

    """Define a layer of the Resneset class"""
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


"""Define function that loads the test data from the prepared data folder
Input arguments: input_folder_path (Path): Path to the test data folder.
Returns: Tuple[torch.tensor, int]: Tuple containing the test images and labels."""
def load_test_data(input_folder_path: Path):

    with open(input_folder_path / "test.pkl",'rb') as test_file:
        X_test,y_test = pickle.load(test_file)
    return X_test, y_test


"""Function to read a model from a zip archive"""
def read_zip(model_path,model_name):

    with zipfile.ZipFile(model_path,'r') as zip_ref:

        if model_name in zip_ref.namelist():
            with zip_ref.open(model_name) as file:
                model_bytes = io.BytesIO(file.read())

            model = torch.load(model_bytes)
        else:
            print(model_name + 'not found in the zip archive.')
    return model




"""Define function that evaluates the model using the test data.
Input arguments:
    model_file_name (str): Filename of the model to be evaluated.
    x (torch.tensor): Test images.
    y (int list): Validation target.
Returns: Accuracy of the model on teh test set """
def evaluate_model(checkpoint_path, loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()


    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

    test_acc = 100 * correct / total
    print('Accuracy of the network on the {} test images: {} %'.format(5000, test_acc))

    return test_acc


if __name__ == "__main__":

    # Load test data
    X_test, y_test = load_test_data(PROCESSED_DATA_DIR / "test")

    # Create a test dataset
    dataset_test = AlzheimerDataset(X_test,y_test)
    testLoader = DataLoader(dataset_test, batch_size=64, shuffle=True)

    # Define the model path and name
    model_path = MODELS_FOLDER_PATH / "alzheimerModel.zip"

    # Evaluate the model on the test data
    test_acc = evaluate_model(
        model_path, testLoader
    )

    # Retrive the content of the file
    with open(METRICS_DIR / "scores.json", 'r') as json_file:
        scores = json.load(json_file)
    
    # Update the dictionary
    scores["test_accuracy"] = test_acc 

    # Update the file scores.json with the new metric
    with open(METRICS_DIR / "scores.json", 'w') as scores_file:
        json.dump(scores, scores_file, indent=4)


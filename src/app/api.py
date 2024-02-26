
import pickle
import numpy as np
import os
from io import BytesIO
from PIL import Image
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import List
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from pathlib import Path
from src.app.schemas import ResNet, ResidualBlock
import zipfile
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

# Define application
app = FastAPI(
    title="Alzheimer's Disease presence classification API",
    description="This API lets you classify the presence of Alzheimer's disease in a brain image",
    version="1.0",
)


def construct_response(f):
    # Decorator for wrapping the response construction logic around other functions
    @wraps(f)
    async def wrap(request: Request, *args, **kwargs):
        # Call the original function, waiting for the results
        results = await f(request, *args, **kwargs)

        # Construct response with required data
        response = {
            "message": results["message"], 
            "method": request.method,       
            "status-code": results["status-code"],  
            "timestamp": datetime.now(),    
            "url": request.url._url,      
        }

        # Add data to the response if available in the results
        if "data" in results:
            response["data"] = results["data"]

        return response  

    return wrap  


@app.on_event("startup")
def _load_models():
    """
    Returns the model with the weights obtained in training
    """
    # Path to the model zip file
    model_path = MODELS_FOLDER_PATH / "alzheimerModelBest.zip"

    # Define the neural network architecture
    model = ResNet(ResidualBlock, [3, 4, 6, 3])

    # Load the pre-trained model weights and set the architecture
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    
    return model 




@app.get("/", tags=["General"])  
@construct_response  # Decorator that wraps the function for constructing the API response
async def _index(request: Request):
    """
    Returns a message to welcome users accessing the API's root.
    """

    response = {
        "message": HTTPStatus.OK.phrase,  # Indicating a successful connection with status code 200
        "status-code": HTTPStatus.OK,
        "data": {
            "message": "Welcome to the Alzheimer's Disease Presence Classifier, also called ADPC. Upload an image of the patient's brain to know the severity of the situation."
        }, 
    }

    return response  




@app.get("/models", tags=["Models"])  
@construct_response
async def get_models(request: Request):
    """
    Returns a response containing information about available models and their associated metrics.
    """
    model_files = os.listdir(MODELS_FOLDER_PATH)
    metrics_files = os.listdir(ROOT_DIR)

    # Filter the model names
    model_names = [file.split(".")[0] for file in model_files if file.endswith(".zip")]

    # Prepare a dictionary to store model names and their respective metrics
    metric_file = "params.yaml"
    with open(os.path.join(ROOT_DIR, metric_file)) as metric_file_content:
        metric_data = yaml.safe_load(metric_file_content)
    with open(os.path.join(METRICS_DIR, "scoresBest.json")) as file_content:
        scores = json.load(file_content)
    # Create a list of model names along with their metrics
    models_info = [{"name": name, "metrics": metric_data, "scores":scores} for name in model_names]

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"models": models_info}
    }
    return response



def get_presence(presence):
    """
    Converts a label ID to its corresponding name.
    """
    if presence[0].item()==0:
        return "Mild Demented"
    if presence[0].item()==1:
        return "Moderate Demented"
    if presence[0].item()==2:
        return "Non Demented"
    if presence[0].item()==3:
        return "Very Mild Demented"
        

@app.post("/models/{type}", tags=["Prediction"])
@construct_response
async def _predict(request : Request, file : UploadFile): 
    """
    Reads an uploaded image and returns the prediction of the model
    """
    image_bytes = await file.read()  # Reading the bytes of the uploaded image file
    stream = BytesIO(image_bytes)  # Creating a stream from the bytes data
    image = Image.open(stream)  # Converting the byte stream to a PIL Image

    # Preprocessing the image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing the image to (224, 224) pixels
        transforms.ToTensor(),  # Converting the PIL Image to a PyTorch tensor
    ])
    image = preprocess(image)  
    image = image.unsqueeze(0)  # Adding an extra dimension to make it a 4D tensor (batch dimension)

    # Loading the neural network model
    model = _load_models() 

    # Performing inference on the processed image
    output = model(image)  
    output = torch.softmax(output, dim=1) 
    probs, idxs = output.topk(1)  
    presence = get_presence(idxs)  

    # Generating the response with the prediction result
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "Alzheimer presence": presence,
        },
    }

    return response 


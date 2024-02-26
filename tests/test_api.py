from http import HTTPStatus
import os
import sys
from io import BytesIO
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import pytest
import io
from src.app.api import app

# Fixture to set up the TestClient instance for the FastAPI app
@pytest.fixture(scope="module", autouse=True)
def client():
    # Create a TestClient instance to trigger the FastAPI startup and shutdown events
    with TestClient(app) as client:
        return client

# Test the root endpoint
def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    # Check the response message for the root endpoint
    assert response.json()["data"]["message"] == "Welcome to the Alzheimer's Disease Presence Classifier, also called ADPC. Upload an image of the patient's brain to know the severity of the situation."

# Test for the get_models endpoint
def test_get_models(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()["data"]
    
    # Perform specific tests based on the expected output structure
    assert "models" in data  # Check the presence of 'models' key in the response data
    assert isinstance(data["models"], list)  # Check if the 'models' key contains a list of dictionaries
    
    # Check if the expected keys are present in the model information
    for model_info in data["models"]:
        assert "name" in model_info
        assert "metrics" in model_info

# Test the model prediction functionality
def test_model_prediction(client):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    image_filename = "image_pil.png"
    image_path = os.path.join(script_dir, image_filename)
    if os.path.isfile(image_path):
        # Simulate uploading an image
        _files = {'uploadFile': open("image_pil.png",'rb')}
        # Send a POST request to the /Prediction endpoint
        response = client.post("/Prediction", files={"file": ("filename", open(image_filename, "rb"), "image/jpeg")})

        assert response.status_code == 200
    else:
        pytest.fail("File does not exist.")

if __name__ == "__main__":
    pytest.main()


# TAED2-ML-Alphas - Image classification for Alzheimer MRI Disease

Alzheimer’s disease is a brain disorder that affects more than 55 million people worldwide according to the World Health Organization. Furthermore, the application of new data driven tools and Artificial Intelligence based solutions to its diagnosis and treatment could make a huge impact on patients suffering from it. For this reason, in our project, we will be aiming at classifying MRI scans to detect Alzheimer’s disease. 

In this context, we propose a fine tuned ResNet to handle the task of image classification applied to MRI scans to detect the severity of the Alzheimer’s disease. 

This project in the "Advanced Data Engineering 2" course aims to create and deploy a Machine Learning component while adhering to software engineering best practices. It involves interpreting key Software Engineering concepts for Machine Learning systems, applying good software engineering practices for data science and machine learning, and implementing MLOps practices to ensure model reproducibility and quality. The project's final goal is to apply MLOps practices for deploying Machine Learning models and promoting API development.

## Project Summary

Our project is summarized in the dataset and model cards, which can be found in:

- [dataset card](https://github.com/MLOps-essi-upc/taed2-ML-Alphas/blob/main/docs/data%20card.md)
- [model card](https://github.com/MLOps-essi-upc/taed2-ML-Alphas/blob/main/docs/model%20card.md)

 ## Project structure
 The project has the following structure (only printed until 3 levels of the tree). It should be noted that we only show the folders, since otherwise the structure tree is too long.
 
     taed2-ML-Alphas/
     ├── data
     │   ├── prepared_data
     │   │   ├── test
     │   │   └── train
     │   ├── raw
     │   ├── raw_data
     │   │   ├── test
     │   │   └── train
     │   └── test
     ├── docs
     ├── great_expectations
     │   ├── checkpoints
     │   ├── expectations
     │   ├── plugins
     │   │   └── custom_data_docs
     │   ├── profilers
     │   └── uncommitted
     │       └── validations
     ├── gx
     │   ├── checkpoints
     │   ├── expectations
     │   ├── plugins
     │   │   └── custom_data_docs
     │   ├── profilers
     │   └── uncommitted
     │       ├── data_docs
     │       └── validations
     ├── metrics
     ├── models
     ├── notebooks
     │   └── __pycache__
     ├── __pycache__
     ├── references
     ├── reports
     │   └── figures
     ├── src
     │   ├── app
     │   │   └── __pycache__
     │   ├── features
     │   └── models
     ├── tests
     │   ├── mlruns
     │   │   └── 0
     │   └── __pycache__
     └── venv
         ├── bin
         └── lib
             └── python3.8


## Getting Started
To set up and run the project, you should run the following commands in your terminal:

- $ git clone https://github.com/yourusername/taed2-ML-Alphas.git
- cd taed2-ML-Alphas
- $ pip freeze > requirements.txt
- dvc repro

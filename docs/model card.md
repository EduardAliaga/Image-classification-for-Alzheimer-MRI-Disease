## Model Card
The model consists of a ResNet architecture, a deep neural network architecture designed to overcome the vanishing gradient problem and improve the training of  deep neural networks by using residual blocks. It is a modified and fined-tunned version of the ResNet-34 from https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/.

### Model Details
#### Model Description
ResNet (Residual Network) is a deep neural network architecture designed to overcome the vanishing gradient problem and improve the training of very deep neural networks.  It uses residual blocks, which introduce skip connections that allow the network to bypass one or more layers. These enable the flow of gradient information during training, making it easier to train deep networks without suffering from diminishing gradients. In each residual block, the input to the block is combined with the output of the block, allowing the network to learn the residual, or the difference between the desired output and the input. 
<img width="958" alt="resnet_architecture" src="https://github.com/MLOps-essi-upc/taed2-ML-Alphas/assets/71087191/9b106c52-c463-4fcc-8a9b-d8d9d69412a7">

Particularly, this ResNet has a stride of 2 for the convolution and a padding of 3. On the other hand, the Residual Block has convolutions of stride and padding of value 1.

- **Developed by:** Gerard Martin
- **Model type:** ResNet-34
- **Language(s):** Python
- **License:** {{ license | default("[More Information Needed]", true)}}
- **Finetuned from model:** https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/

### Intended Uses
- **Primary intended use:** Classify brain pictures of MRI scans according to the severity of the Alzheimer disease they present.
- **Primary intended users:** Researchers and health medicine applications.
- **Out-of-scope use cases:** Detection of other brain diseases different from Alzheimer.

### Bias, Risks, and Limitations
The training dataset has a distribution of 50.1% of Non_Demented cases, 34.8% of Very_Mild_Demented, 14.1% of Mild_Demented and 1% of Moderate_Demented. This imbalance on the class distribution may potentially lead to some biases and risks:

- **Bias in model training:** the model may perform well on the majority class ("Non_Demented" one) but poorly on minority classes.If the "Non_Demented" class is not handled carefully, model might have a bias towards classifying instances as "Non_Demented", even if they belong to another.
- **Loss of information:** insufficient training data for minority classes, probably leading to an inadequate understanding.
- **Reduced generalization:** the model may not generalize well to new data, specially for minority classes.
- **Model Overfitting:** the model may overfit the majority class.
- **Ethical Concerns:** biased predictions due to the class imbalance can have ethical implications, especially considering that our model predictions impact individuals' lives.

#### Recommendations
Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.

### How to Get Started with the Model
Use the code below to get started with the model. For the code to work, there are some prerequisites: having Python, PIL (Python Imaging Library), MLflow, NumPy and Pytorch installed.

### Training Details
#### Training data
The training dataset has 5120 rows and a number of bytes of 22,560,791.2. Additionally, the training set has a label distribution of 50.1% of Non_Demented cases, 34.8% of Very_Mild_Demented, 14.1% of Mild_Demented and 1% of Moderate_Demented. 

To prepare this dataset, we resized the image to (224, 224) and converted the resulting PIL image to tensor. After that, the *data_loader* function is called to split the data into training and validation sets. This function starts by normalizing the given dataset. The dataset is split into training and validation sets using the default valid_size:10% of the data will be used for validation.

#### Training Procedure  
First, a GPU is recommended for faster training.

The model can be configured by modifying the variables *num_classes*, *num_epochs*, *batch_size*, and *learning_rate*. By default, the model is trained with the Stochastic Gradient Descent with Momentum optimizer and with the Cross Entropy Loss.  

##### Hyperparameters
The initial training was performed with:
- Number of epochs: 7
- Batch size: 16
- Learning rate: 0.01
- Optimizer: Stochastic Gradient Descent with Momentum
- Loss Function: Cross Entropy Loss
- Regularization: weight decay of 0.001

### Evaluation
#### Testing data
The test dataset contains 1,280 examples and 5,637,447.08 bytes. Nevertheless, its distribution is the following: 49.5% of Non_Demented cases, 35.9% of Very_Mild_Demented, 13.4% Mild_Demented and 1.2% of Moderate_Demented.

#### Metrics
We will use accuracy. Nevertheless, it would be interesting to also analyze metrics such as precision, recall, F1-score and area under the ROC curve (AUC-ROC) in order to appropriately deal with the imbalanced dataset. This will be done in the following sessions.

#### Results

### Environmental Impact
Carbon emissions are estimated using *CodeCarbon*.
- Duration of the compute (in seconds): 17029
- Emissions as CO2-equivalents (in kg): 0.1161
- Energy consumed (in kWh): 0.2565

It should be noted this results are for a training with 15 epochs and a batch size of 32 (which obtains an accuracy of 96.875%).

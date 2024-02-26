from datasets import load_dataset

# Load the 'train' split of the Falah/Alzheimer_MRI dataset
train_dataset = load_dataset('Falah/Alzheimer_MRI', split='train')

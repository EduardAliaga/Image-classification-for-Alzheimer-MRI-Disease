import pickle
from pathlib import Path
import great_expectations as ge
import pandas as pd

# Define path of the root
ROOT_DIR = Path(Path(__file__).resolve().parent.parent).parent

# Define path to the processed data folder
PROCESSED_DATA_DIR = ROOT_DIR / "data/prepared_data"

# Load data from a pickle file (suppose 'train' contains tuples of tensors and labels)
with open(PROCESSED_DATA_DIR / 'train/train.pkl', 'rb') as file:
    tr = pickle.load(file)

# Extract labels and create a dictionary with 'image' and 'label' keys
labels = [item for item in tr[1]]
data_dict = {
    'image': tr[0],
    'label': labels
}

# Create a Pandas DataFrame from the dictionary
train = pd.DataFrame(data_dict)

# Initialize Great Expectations context
context = ge.get_context()

# # Add or update an expectation suite named "alzheimer_training_suite"
context.add_or_update_expectation_suite("alzheimer_training_suite")

# Add data to the data context as a Pandas dataframe
datasource = context.sources.add_or_update_pandas(name="alzheimer_dataset")
data_asset = datasource.add_dataframe_asset(name="training", dataframe=train)

# Define expectations
batch_request = data_asset.build_batch_request()
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="alzheimer_training_suite",
    datasource_name="alzheimer_dataset",
    data_asset_name="training",
)

# Define expectations for column structure and data quality
validator.expect_table_columns_to_match_ordered_list(
    column_list=[
        "image",
        "label"
    ]
)

validator.expect_column_values_to_be_unique("image")

validator.expect_column_values_to_not_be_null("image")

validator.expect_column_values_to_not_be_null("label")

validator.expect_column_values_to_be_of_type("label", "int64")

validator.expect_column_values_to_be_between("label", min_value=0, max_value=3)

validator.save_expectation_suite(discard_failed_expectations=False)

# Save expectation suite
validator.save_expectation_suite(discard_failed_expectations=False)

# Create checkpoint for validations
checkpoint = context.add_or_update_checkpoint(
    name="my_checkpoint",
    validator=validator,
)

# Run validations and view results
checkpoint_result = checkpoint.run()
context.view_validation_result(checkpoint_result)


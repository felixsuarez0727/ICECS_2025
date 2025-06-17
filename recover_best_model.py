import os
import json
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from src.data_loader import DataLoader
import run_main_nas

# Path to the dataset used in the experiment
DATASET_PATH = '/Users/jaimearevalo/Downloads/RadComOta2.45GHz.hdf5'

# Load data to get input_shape and num_classes
loader = DataLoader(DATASET_PATH, combine_am=True, samples_per_class=5000)
X_train, X_val, X_test, y_train, y_val, y_test = loader.load_data()
input_shape = X_train.shape[1:]
num_classes = len(set(y_train))

# Update MODEL_CONFIG globally
run_main_nas.MODEL_CONFIG = {
    'input_shape': input_shape,
    'num_classes': num_classes
}

# Load hyperparameters from trial 17
trial_json = 'results2/nas/nas_search/trial_17/trial.json'
with open(trial_json, 'r') as f:
    trial_data = json.load(f)
hp_values = trial_data['hyperparameters']['values']

# Create HyperParameters object and assign values
hp = kt.HyperParameters()
for k, v in hp_values.items():
    hp.values[k] = v

# Rebuild the model using the build_model function from the run_main_nas module
model = run_main_nas.build_model(hp)

# Load weights
weights_path = 'results2/nas/nas_search/trial_17/checkpoint.weights.h5'
model.load_weights(weights_path)

# Save the complete model
output_path = 'results2/nas/best_model_recovered.h5'
model.save(output_path)
print(f"Recovered model saved at {output_path}")

# Show summary
model.summary()

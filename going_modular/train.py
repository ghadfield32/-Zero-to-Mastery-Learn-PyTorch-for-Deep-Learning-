
"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import argparse
import torch
from torchvision import transforms
import data_setup, engine, model_builder, utils
import json
from model_builder import *  # Import all models
import inspect
import torch

def get_model_list():
    model_list = []
    for name, obj in inspect.getmembers(model_builder):
        if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
            model_list.append(name)
    return model_list

model_choices = get_model_list()

test_dir_default = "data/pizza_steak_sushi/test"
train_dir_default = "data/pizza_steak_sushi/train"
epochs_default = 5
batch_size_default = 32
hidden_units_default = 10
lr_default = 0.001
model_type_default = "TinyVGG"
model_choices_default = model_choices

# Set up the argument parser
parser = argparse.ArgumentParser(description="Train a PyTorch image classification model.")
parser.add_argument("--train_dir", type=str, default=train_dir_default, help="Directory for training data")
parser.add_argument("--test_dir", type=str, default=test_dir_default, help="Directory for testing data")
parser.add_argument("--epochs", type=int, default=epochs_default, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=batch_size_default, help="Batch size for training/testing")
parser.add_argument("--hidden_units", type=int, default=hidden_units_default, help="Number of hidden units in the model")
parser.add_argument("--learning_rate", type=float, default=lr_default, help="Learning rate for the optimizer")
parser.add_argument("--model_type", type=str, default=model_type_default, choices=model_choices, help="Type of model to train")

# Parse arguments
args = parser.parse_args()

# Use arguments
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
train_dir = args.train_dir
test_dir = args.test_dir
MODEL_TYPE = args.model_type

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

os.makedirs("models", exist_ok=True)
# Save class names
class_names_path = "models/class_names.json"  # Adjust the path as needed
with open(class_names_path, "w") as f:
    json.dump(class_names, f)

# Create model based on the selected type
if MODEL_TYPE == "TinyVGG":
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)
elif MODEL_TYPE == "ModelX":
    # Example for another model, replace with actual model code
    model = model_builder.ModelX(
        some_parameter=123,  # Replace with actual parameters
        another_parameter=456
    ).to(device)
else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}")

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

os.makedirs("models", exist_ok=True)
# Save the model state dictionary
model_state_dict_path = "models/05_going_modular_script_mode_tinyvgg_model_state_dict.pth"
torch.save(model.state_dict(), model_state_dict_path)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")

os.makedirs("hyperparameters", exist_ok=True)
# Save hyperparameters
hyperparams = {
    "model_type": MODEL_TYPE,
    "hidden_units": HIDDEN_UNITS,
    # Add other relevant hyperparameters here
}
with open("hyperparameters/hyperparams.json", "w") as f:
    json.dump(hyperparams, f)


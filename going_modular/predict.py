"""
Predict on custom images
"""
import sys
sys.path.append('/content/going_modular')

import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import data_setup, engine, model_builder, utils
from model_builder import * # Adjust the import statement as necessary
import inspect
import json

#Model list
def get_model_list():
    model_list = []
    for name, obj in inspect.getmembers(model_builder):
        if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
            model_list.append(name)
    return model_list

model_choices = get_model_list()

# Load class names
class_names_path = "models/class_names.json"  # Adjust the path as needed
with open(class_names_path, "r") as f:
    class_names = json.load(f)

model_type_default = 'TinyVGG'  # Replace with your default model type
class_names_default = class_names
hidden_units_default = 10
device_default = "cuda" if torch.cuda.is_available() else "cpu"

hyperparams_path = "/content/hyperparameters/hyperparams.json"
if os.path.exists(hyperparams_path):
    with open(hyperparams_path, "r") as f:
        hyperparams = json.load(f)
    model_type_hyperparam = hyperparams.get("model_type", model_type_default)
    hidden_units_hyperparam = hyperparams.get("hidden_units", hidden_units_default)
else:
    model_type_hyperparam = model_type_default
    hidden_units_hyperparam = hidden_units_default


# Set up argument parsing
parser = argparse.ArgumentParser(description="Predict on custom images using a trained model.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
parser.add_argument("--image_path", type=str, required=True, help="Path to the image for prediction.")
parser.add_argument("--class_names", nargs='+', default=class_names_default, help="List of class names.")
parser.add_argument("--device", type=str, default=device_default, help="Compute device to use.")
parser.add_argument("--hidden_units", type=int, default=hidden_units_hyperparam, help="Number of hidden units in the model")
parser.add_argument("--model_type", type=str, default=model_type_hyperparam, choices=model_choices, help="Type of model to train")

args = parser.parse_args()
device = torch.device(args.device)

def load_model(model_type, hidden_units, model_path, device):
    # Load model based on type and parameters
    # For example:
    if model_type == "TinyVGG":
        model = TinyVGG(input_shape=3, hidden_units=hidden_units, output_shape=len(class_names))
    elif model_type == "ModelX":
        model = ModelX(some_parameter=123)  # Adjust accordingly
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

model = load_model(args.model_type, args.hidden_units, args.model_path, device)

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Adjust based on the input size used during training
    transforms.ToTensor()
])

def load_and_transform_image(image_path, transform):
    image = Image.open(image_path)
    image = transform(image)
    return image

image_tensor = load_and_transform_image(args.image_path, transform)

def pred_and_plot_image(model: torch.nn.Module,
                        image_tensor: torch.Tensor,
                        device: torch.device,
                        class_names: list = class_names):
    """
    Makes a prediction on a target image and plots the image with its prediction.
    Args:
    model: The trained PyTorch model.
    image_tensor: The image tensor to make a prediction on.
    class_names: Optional list of class names for the model's output.
    device: The device to perform computation (e.g., 'cuda' or 'cpu').

    Ex usage:
    # Pred on our custom image
    pred_and_plot_image(model=model_1,
                    image_tensor=image_tensor,
                    class_names=class_names,
                    device=device)
    """
    # 1. Make sure the model is on the target device
    model.to(device)

    # 2. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image tensor
        target_image_tensor = image_tensor.unsqueeze(dim=0).to(device)

        # Make a prediction on the image tensor
        target_image_pred = model(target_image_tensor)

    # 3. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 4. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 5. Plot the image alongside the prediction and prediction probability
    plt.imshow(image_tensor.permute(1, 2, 0)) # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False);
    # Save the plot to a file
    plt.savefig("prediction_output.png")

    # Show the plot (this might not work in all environments)
    plt.show()

# Load and transform image
image = load_and_transform_image(args.image_path, transform)

# At the bottom of your script, where the function call is made
pred_and_plot_image(model, image_tensor, device, args.class_names or class_names_default)


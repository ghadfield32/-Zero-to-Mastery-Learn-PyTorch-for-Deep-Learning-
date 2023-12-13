"""
Contains functionality for creating PyTorch DataLoaders for
image classification data with automated normalization.
"""
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def calculate_mean_std(loader):
    """Calculate the mean and standard deviation of a dataset."""
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    return mean.numpy(), std.numpy()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    image_size: tuple = (224, 224),
    batch_size: int = 32,
    num_workers: int = NUM_WORKERS
):
    """Creates training and testing DataLoaders with automatic normalization.
  
    Args:
      train_dir: Path to training directory.
      test_dir: Path to testing directory.
      image_size: Size to resize images (height, width).
      batch_size: Number of samples per batch in each DataLoader.
      num_workers: Number of workers per DataLoader.
  
    Returns:
      A tuple of (train_dataloader, test_dataloader, class_names).
    """
    # Initial transform to calculate normalization stats
    initial_transform = transforms.Compose([
        transforms.Resize(image_size), 
        transforms.ToTensor()
    ])
  
    # Calculate mean and std for normalization
    temp_dataset = datasets.ImageFolder(train_dir, transform=initial_transform)
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    mean, std = calculate_mean_std(temp_loader)

    # Define transformation with normalization
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Create training and testing datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,  # don't need to shuffle test data
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names


#num_workers = 1
#batch_size = 32

# Example usage:
#train_dataloader, test_dataloader, class_names = create_dataloaders(
#     train_dir=train_dir,
#     test_dir=test_dir,
#     image_size=(224, 224),
#     batch_size=batch_size,
#     num_workers=num_workers
# )



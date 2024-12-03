import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os
from collections import defaultdict

# Import the ConvNet architecture from part2
from part2 import ConvNet

def load_variable_res_dataset(root_dir):
    """
    Load images while preserving their original resolution,
    grouping them by resolution.
    """
    resolution_groups = defaultdict(list)
    labels = []
    
    # Transform for converting to grayscale and tensor, but not resizing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    
    # Walk through all class directories
    for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Process each image in the class directory
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path)
            resolution = img.size  # (width, height)
            
            # Transform image
            img_tensor = transform(img)
            
            # Store image tensor and label, grouped by resolution
            resolution_groups[resolution].append(img_tensor)
            labels.append(class_idx)
    
    # Convert lists to tensors for each resolution group
    for res in resolution_groups:
        resolution_groups[res] = torch.stack(resolution_groups[res])
    
    labels = torch.tensor(labels)
    return resolution_groups, labels

def create_batches(resolution_groups, labels, batch_size):
    """
    Create batches of uniform resolution images.
    Returns a list of (batch_images, batch_labels) tuples.
    """
    batches = []
    
    for resolution, images in resolution_groups.items():
        # Get indices of images with this resolution
        indices = torch.where(torch.tensor([img.shape[-2:] == images[0].shape[-2:] 
                                          for img in images]))[0]
        
        # Create batches for this resolution group
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_images = images[batch_indices]
            batch_labels = labels[batch_indices]
            batches.append((batch_images, batch_labels))
    
    return batches

if __name__ == "__main__":
    # Setup
    BATCH_SIZE = 16
    EPOCHS = 8
    LEARNING_RATE = 0.001
    
    # Load datasets
    train_groups, train_labels = load_variable_res_dataset('mnist-varres/train')
    test_groups, test_labels = load_variable_res_dataset('mnist-varres/test')
    
    # Create batches
    train_batches = create_batches(train_groups, train_labels, BATCH_SIZE)
    test_batches = create_batches(test_groups, test_labels, BATCH_SIZE)

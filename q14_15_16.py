import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os
from collections import defaultdict

class FlexibleConvNet(nn.Module):
    def __init__(self, num_classes=10, N=64, pooling='max'):
        super(FlexibleConvNet, self).__init__()
        
        # Determine pooling type
        if pooling.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size=2)
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        elif pooling.lower() == 'mean':
            self.pool = nn.AvgPool2d(kernel_size=2)
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError("Pooling must be either 'max' or 'mean'")
        
        # First conv block
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        # Second conv block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
        # Third conv block
        self.conv3 = nn.Conv2d(32, N, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        # Final classification layer
        self.fc = nn.Linear(N, num_classes)
    
    def forward(self, x):
        # Conv blocks
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = self.pool(self.relu3(self.conv3(x)))
        
        # Global pooling and flatten
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.fc(x)
        return x

def load_resolution_datasets(root_dir):
    """
    Load images into three separate tensors based on resolution.
    Returns a dictionary mapping resolution tuples to (images, labels) pairs.
    """
    datasets = defaultdict(list)
    resolution_labels = defaultdict(list)
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    
    for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path)
            resolution = img.size  # (width, height)
            
            img_tensor = transform(img)
            datasets[resolution].append(img_tensor)
            resolution_labels[resolution].append(class_idx)
    
    # Convert to tensors
    resolution_datasets = {}
    for res in datasets:
        images = torch.stack(datasets[res])
        labels = torch.tensor(resolution_labels[res])
        resolution_datasets[res] = (images, labels)
    
    return resolution_datasets

def train_epoch(model, resolution_datasets, optimizer, criterion, batch_size, device):
    model.train()
    total_loss = 0
    total_samples = 0
    
    # Train on each resolution group
    for resolution, (images, labels) in resolution_datasets.items():
        # Create batches for this resolution
        num_samples = images.size(0)
        indices = torch.randperm(num_samples)
        
        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_x = images[batch_indices].to(device)
            batch_y = labels[batch_indices].to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch_indices)
            total_samples += len(batch_indices)
    
    return total_loss / total_samples

def evaluate(model, resolution_datasets, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for resolution, (images, labels) in resolution_datasets.items():
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    # Training parameters
    BATCH_SIZE = 16
    EPOCHS = 8
    LEARNING_RATE = 0.001
    
    # Load datasets
    train_datasets = load_resolution_datasets('mnist-varres/train')
    test_datasets = load_resolution_datasets('mnist-varres/test')
    
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlexibleConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print('Epoch 0 (Before Training):')
    train_acc = evaluate(model, train_datasets, device)
    print(f'Training Accuracy: {train_acc:.4f}')
    print('-' * 50)
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_datasets, optimizer, criterion, 
                               BATCH_SIZE, device)
        train_acc = evaluate(model, train_datasets, device)
        
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Training Accuracy: {train_acc:.4f}')
        print('-' * 50)
    
    # Final evaluation
    test_acc = evaluate(model, test_datasets, device)
    print(f'Final Test Accuracy: {test_acc:.4f}')
    
    model = FlexibleConvNet(N=64)
    print(f"Parameters with N=64: {count_parameters(model)}")
    model = FlexibleConvNet(N=128)
    print(f"Parameters with N=128: {count_parameters(model)}")
    model = FlexibleConvNet(N=256)
    print(f"Parameters with N=256: {count_parameters(model)}")


# For handling datasets where almost every image has a unique resolution, here are some potential approaches:
# Batch Similar Resolutions: Group images with similar resolutions together and pad them to the largest size in the group.
# or use batch_size=1 


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from collections import defaultdict
import matplotlib.pyplot as plt

# Setup
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.003
RESOLUTIONS = [32, 48, 64]

class AdaptiveConvNet(nn.Module):
    def __init__(self, N=16):
        super(AdaptiveConvNet, self).__init__()
        
        # First block: 1 -> 16 channels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Second block: 16 -> 32 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Third block: 32 -> N channels
        self.conv3 = nn.Conv2d(32, N, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Global pooling (can use either max or mean)
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        
        # Final linear layer
        self.fc = nn.Linear(N, 10)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        x = self.global_pool(x)
        x = x.view(-1, x.size(1))
        x = self.fc(x)
        return x

def group_by_resolution(data, labels):
    """Group data and labels by image resolution"""
    groups = defaultdict(list)
    for img, label in zip(data, labels):
        h, w = img.shape[1:]
        groups[h].append((img, label))
    
    # Convert lists to tensors
    grouped_data = {}
    for res, img_label_list in groups.items():
        images = torch.stack([pair[0] for pair in img_label_list])
        labels = torch.tensor([pair[1] for pair in img_label_list], dtype=torch.long)
        grouped_data[res] = (images, labels)
    
    return grouped_data

def train_epoch(model, grouped_data, optimizer, criterion, batch_size, device):
    model.train()
    total_loss = 0
    total_batches = 0
    
    # Train on each resolution group separately
    for resolution, (data, labels) in grouped_data.items():
        # Shuffle data
        indices = torch.randperm(len(data))
        data = data[indices]
        labels = labels[indices]
        
        for i in range(0, len(data), batch_size):
            batch_x = data[i:i+batch_size].to(device)
            batch_y = labels[i:i+batch_size].to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
        
    return total_loss / total_batches if total_batches > 0 else float('inf')

def evaluate(model, grouped_data, batch_size, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for resolution, (data, labels) in grouped_data.items():
            for i in range(0, len(data), batch_size):
                batch_x = data[i:i+batch_size].to(device)
                batch_y = labels[i:i+batch_size].to(device)
                
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
    
    return correct / total if total > 0 else 0

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Basic transform - just convert to tensor and grayscale
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
    ])
    
    # Load your dataset from the specified directories
    train_dataset = torchvision.datasets.ImageFolder(root='mnist-varres/train', transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root='mnist-varres/test', transform=transform)
    
    # Extract data and labels
    train_data, train_labels = zip(*[(data, label) for data, label in train_dataset])
    test_data, test_labels = zip(*[(data, label) for data, label in test_dataset])
    
    # Check data loading
    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of test samples: {len(test_data)}")
    print(f"Sample image shape: {train_data[0].shape}")
    print(f"Sample label: {train_labels[0]}")
    
    # Group by resolution using the existing function
    grouped_train_data = group_by_resolution(train_data, train_labels)
    grouped_test_data = group_by_resolution(test_data, test_labels)
    print(f"Sample image min: {train_data[0].min()}, max: {train_data[0].max()}")
    print(f"Class to index mapping: {train_dataset.class_to_idx}")
    
    # Initialize model, loss function, and optimizer
    model = AdaptiveConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    
    # Training loop
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, grouped_train_data, optimizer, criterion, BATCH_SIZE, device)
        
        # Update learning rate based on loss
        scheduler.step(train_loss)
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss:.4f}, LR: {current_lr:.6f}')
        
        # Optional: Print accuracy after each epoch
        train_acc = evaluate(model, grouped_train_data, BATCH_SIZE, device)
        test_acc = evaluate(model, grouped_test_data, BATCH_SIZE, device)
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc*100:.2f}%, Test Acc: {test_acc*100:.2f}%')
    
    # Final evaluation
    final_accuracy = evaluate(model, grouped_test_data, BATCH_SIZE, device)
    print(f'Final Test Accuracy: {final_accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

# Setup constants
BATCH_SIZE = 35
EPOCHS = 14
LEARNING_RATE = 0.0009528713951024067



# # Define transforms for training (with augmentation)
# train_transforms = transforms.Compose([
#     transforms.RandomRotation(15),
#     transforms.RandomAffine(
#         degrees=0,
#         translate=(0.1, 0.1),
#         scale=(0.9, 1.1)
#     ),
#     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
#     transforms.ToTensor(),
# ])

# train_transforms = transforms.Compose([
#     transforms.RandomRotation(15),
#     transforms.RandomAffine(
#         degrees=0,
#         translate=(0.1, 0.1),
#         scale=(0.9, 1.1)
#     ),
#     transforms.ToTensor(),
#     transforms.RandomErasing(p=0.5, scale=(0.02, 0.15)),  # Randomly erase parts
#     transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)  # Add Gaussian noise
# ])

#FROM Kaggle
train_transforms = transforms.Compose([
    transforms.RandomRotation(18),  # 18 degrees rotation
    transforms.RandomAffine(
        degrees=0,
        scale=(0.8, 1.2),  # zoom range of 0.2 (20%) both in and out
        fill=0  # fill value for empty pixels
    ),
    transforms.ToTensor(),
])

# No augmentations for validation and test sets
val_transforms = transforms.Compose([
    transforms.ToTensor()
])

# Load the full training dataset with augmentation transforms
full_train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True,
    download=True, 
    transform=train_transforms
)

# Split into train and validation sets
train_size = 50000
val_size = 10000
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Load test dataset
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False,
    download=True, 
    transform=val_transforms
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=2
)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Second conv block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Third conv block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected layer
        self.fc = nn.Linear(64 * 3 * 3, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = x.view(-1, 64 * 3 * 3)
        x = self.fc(x)
        return x

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return correct / total

if __name__ == "__main__":
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initial evaluation
    print('Epoch 0 (Before Training):')
    train_acc = evaluate(model, train_loader, device)
    val_acc = evaluate(model, val_loader, device)
    print(f'Training Accuracy: {train_acc:.4f}')
    print(f'Validation Accuracy: {val_acc:.4f}')
    print('-' * 50)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Iterate over batches using DataLoader
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate
        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)
        
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'Training Loss: {total_loss / len(train_loader):.4f}')
        print(f'Training Accuracy: {train_acc:.4f}')
        print(f'Validation Accuracy: {val_acc:.4f}')
        print('-' * 50)

    # Final test evaluation
    test_acc = evaluate(model, test_loader, device)
    print(f'Final Test Accuracy: {test_acc:.4f}')
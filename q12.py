import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Import the ConvNet architecture from part2
from part2 import ConvNet, evaluate, train_epoch

# Setup
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.001

# Define transforms
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize all images to 28x28
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
])

# Load datasets using ImageFolder
train_dataset = ImageFolder(root='mnist-varres/train', transform=transform)
test_dataset = ImageFolder(root='mnist-varres/test', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def train_epoch_loader(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate_loader(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
    return correct / total

if __name__ == "__main__":
    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print('Epoch 0 (Before Training):')
    train_acc = evaluate_loader(model, train_loader, device)
    print(f'Training Accuracy: {train_acc:.4f}')
    print('-' * 50)

    for epoch in range(EPOCHS):
        train_loss = train_epoch_loader(model, train_loader, optimizer, criterion, device)
        train_acc = evaluate_loader(model, train_loader, device)
        
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Training Accuracy: {train_acc:.4f}')
        print('-' * 50)

    # Final test evaluation
    test_acc = evaluate_loader(model, test_loader, device)
    print(f'Final Test Accuracy: {test_acc:.4f}')

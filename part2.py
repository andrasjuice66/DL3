import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

# Setup
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.001

# Define the CNN architecture
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
        # First block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Second block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Third block
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Flatten and fully connected
        x = x.view(-1, 64 * 3 * 3)
        x = self.fc(x)
        return x

# Load and prepare data
full_train = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transforms.ToTensor())
test = torchvision.datasets.MNIST(root='./data', train=False,
                                download=True, transform=transforms.ToTensor())

# Split into train and validation (50000 and 10000)
train_dataset, val_dataset = random_split(full_train, [50000, 10000])

# Convert to tensors (kept on CPU)
train_data = torch.stack([x[0] for x in train_dataset])
train_labels = torch.tensor([x[1] for x in train_dataset])
val_data = torch.stack([x[0] for x in val_dataset])
val_labels = torch.tensor([x[1] for x in val_dataset])
test_data = torch.stack([x[0] for x in test])
test_labels = torch.tensor([x[1] for x in test])

def train_epoch(model, train_data, train_labels, optimizer, criterion, batch_size, device):
    model.train()
    total_loss = 0
    
    # Loop over batches
    for i in range(0, len(train_data), batch_size):
        # Get batch
        batch_x = train_data[i:i+batch_size].to(device)
        batch_y = train_labels[i:i+batch_size].to(device)
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / (len(train_data) // batch_size)

def evaluate(model, data, labels, batch_size, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch_x = data[i:i+batch_size].to(device)
            batch_y = labels[i:i+batch_size].to(device)
            
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
    train_acc = evaluate(model, train_data, train_labels, BATCH_SIZE, device)
    val_acc = evaluate(model, val_data, val_labels, BATCH_SIZE, device)
    print(f'Training Accuracy: {train_acc:.4f}')
    print(f'Validation Accuracy: {val_acc:.4f}')
    print('-' * 50)

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_data, train_labels, optimizer, criterion, BATCH_SIZE, device)
        train_acc = evaluate(model, train_data, train_labels, BATCH_SIZE, device)
        val_acc = evaluate(model, val_data, val_labels, BATCH_SIZE, device)
        
        print(f'Epoch {epoch+1}/{EPOCHS}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Training Accuracy: {train_acc:.4f}')
        print(f'Validation Accuracy: {val_acc:.4f}')
        print('-' * 50)

    # Final test evaluation
    test_acc = evaluate(model, test_data, test_labels, BATCH_SIZE, device)
    print(f'Final Test Accuracy: {test_acc:.4f}')

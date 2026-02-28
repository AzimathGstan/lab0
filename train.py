import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import SimpleMNIST # The model we built earlier

def train():
    # 1. Setup: Force CPU to avoid driver nightmares
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Data Loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST mean/std
    ])

    print("Loading dataset...")
    # download=True acts as a failsafe, but it will use our bundled files
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 3. Model, Loss, Optimizer
    model = SimpleMNIST().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop (Keep it to 3 epochs for a fast lab)
    epochs = 3
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        # tqdm gives us a nice progress bar in the terminal
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    # 5. Save the weights
    import os
    os.makedirs('weights', exist_ok=True)
    torch.save(model.state_dict(), 'weights/mnist_model.pth')
    print("\nTraining complete! Weights saved to weights/mnist_model.pth")

if __name__ == '__main__':
    train()

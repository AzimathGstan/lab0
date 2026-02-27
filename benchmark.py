import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import SimpleMNIST

def benchmark():
    device = torch.device('cpu')
    print("Loading test dataset for benchmarking...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = SimpleMNIST().to(device)
    try:
        model.load_state_dict(torch.load('weights/mnist_model.pth', map_location=device, weights_only=True))
        model.eval()
    except FileNotFoundError:
        print("Error: Train the model first using train.py!")
        return

    correct = 0
    total = len(test_dataset)
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / total
    print(f"\nBenchmark Complete!")
    print(f"Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    
    if accuracy < 90.0:
        print("Hint: Try adding more convolutional filters or an extra layer in model.py!")
    elif accuracy > 98.0:
        print("Excellent! You've built a highly accurate model.")

if __name__ == '__main__':
    benchmark()

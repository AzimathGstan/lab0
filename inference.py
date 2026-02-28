import torch
import random
from torchvision import datasets, transforms
from model import SimpleMNIST

def print_ascii_image(image_tensor):
    # ASCII density characters mapping from light (empty) to dark (filled)
    chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']
    
    print("\n--- MNIST Image ---")
    # image_tensor is 28x28 with raw values from 0 to 255
    for row in image_tensor:
        line = ""
        for pixel in row:
            # Map the 0-255 pixel value to an index between 0 and 9
            char_idx = int((pixel.item() / 255.0) * 9)
            # Print each character twice to fix terminal font aspect ratios 
            # (characters are usually twice as tall as they are wide)
            line += chars[char_idx] * 2 
        print(line)
    print("-------------------\n")

def infer():
    # Force CPU to match training environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the model architecture and weights
    model = SimpleMNIST().to(device)
    try:
        model.load_state_dict(torch.load('weights/mnist_model.pth', map_location=device, weights_only=True))
        model.eval() # Set to evaluation mode
    except FileNotFoundError:
        print("Error: Could not find 'weights/mnist_model.pth'. Make sure train.py finished successfully!")
        return

    # 2. Setup the test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # download=False because we bundled the data!
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    
    # 3. Pick a random test image
    idx = random.randint(0, len(test_dataset) - 1)
    image, true_label = test_dataset[idx]
    
    # Grab the raw, un-normalized 0-255 image for the ASCII printer
    raw_image = test_dataset.data[idx] 
    
    # 4. Run the forward pass
    with torch.no_grad():
        # Add a batch dimension: [1, 28, 28] becomes [1, 1, 28, 28]
        output = model(image.unsqueeze(0))
        # Get the index of the highest log-probability
        prediction = output.argmax(dim=1, keepdim=True).item()

    # 5. Output the results
    print_ascii_image(raw_image)
    print(f"True Label: {true_label}")
    print(f"Prediction: {prediction}")
    
    if true_label == prediction:
        print("Result: CORRECT! 🎉")
    else:
        print("Result: INCORRECT ❌")

if __name__ == '__main__':
    infer()

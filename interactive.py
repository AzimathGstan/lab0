import cv2
import numpy as np
import torch
from torchvision import transforms
from model import SimpleMNIST

# Canvas setup
CANVAS_SIZE = 280
BRUSH_RADIUS = 12

drawing = False
canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)

def draw(event, x, y, flags, param):
    global drawing, canvas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(canvas, (x, y), BRUSH_RADIUS, 255, -1)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), BRUSH_RADIUS, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def predict_digit(model, image, device):
    # 1. Resize the 280x280 canvas down to 28x28
    resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 2. Apply the exact same transformations used in training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Convert numpy array to PIL Image format expected by ToTensor
    tensor_img = transform(resized).unsqueeze(0).to(device)
    
    # 3. Predict
    with torch.no_grad():
        output = model(tensor_img)
        prediction = output.argmax(dim=1, keepdim=True).item()
        confidence = torch.exp(output).max().item() * 100
        
    return prediction, confidence, resized

def main():
    global canvas
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleMNIST().to(device)
    try:
        model.load_state_dict(torch.load('weights/mnist_model.pth', map_location=device, weights_only=True))
        model.eval()
    except FileNotFoundError:
        print("Error: Could not find weights. Run train.py first!")
        return

    cv2.namedWindow('Draw a Digit')
    cv2.setMouseCallback('Draw a Digit', draw)

    print("\n--- Interactive MNIST ---")
    print("Draw a number (0-9) in the window.")
    print("Press 'Space' to predict.")
    print("Press 'c' to clear the canvas.")
    print("Press 'q' to quit.")

    while True:
        cv2.imshow('Draw a Digit', canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
        elif key == 32: # Spacebar
            pred, conf, downscaled = predict_digit(model, canvas, device)
            print(f"Prediction: {pred} (Confidence: {conf:.2f}%)")
            
            # Show what the model actually sees
            cv2.imshow('28x28 Model Input', cv2.resize(downscaled, (140, 140), interpolation=cv2.INTER_NEAREST))

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

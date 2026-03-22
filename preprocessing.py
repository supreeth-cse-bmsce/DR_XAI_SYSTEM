import cv2
import torch
from torchvision import transforms

def preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224,224))
    image = image.astype("float32") / 255.0   # <-- Important fix

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    tensor = transform(image).unsqueeze(0).float()   # <-- Force float32
    return tensor
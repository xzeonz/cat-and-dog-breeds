import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import io
import os
from typing import cast

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
NUM_CLASSES = 55
class_names = [
    'Abyssinian', 'Alaskan_Malamute', 'American_Bobtail', 'American_Shorhair', 'American_bulldog',
    'American_pit_bull_terrier', 'Basset_hound', 'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer',
    'British_shorthair', 'Bulldog', 'Calico', 'Chihuahua', 'Dachshund', 'Egyptian_mau',
    'English_cocker_paniel', 'English_setter', 'German_Shepherd', 'German_shorthairaired',
    'Golden_Retreiver', 'Great_pyrenees', 'Havanese', 'Husky', 'Japanese_chin', 'Keeshond',
    'Labrador_Retriever', 'Leonberger', 'Maine_coon', 'Miniature_pinscher', 'Munchkin',
    'Newfoundland', 'Norwegian_Forest_Cat', 'Ocicat', 'Persian', 'Pomeranian', 'Poodle', 'Pug',
    'Ragdoll', 'Rottweiler', 'Russian_blue', 'Saint_bernard', 'Samoyed', 'Scottish_Fold',
    'Scottish_terrier', 'Shiba_inu', 'Siamese', 'Sphynx', 'Staffordshire_bull_terrier',
    'Tortoiseshell', 'Tuxedo', 'Wheaten_tersier', 'Yorkshire_terrier'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations (must match your training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    model = models.efficientnet_b0(pretrained=False)
    linear_layer = model.classifier[1]
    num_ftrs: int = cast(int, linear_layer.in_features)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    )
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor_intermediate = cast(torch.Tensor, transform(image))
    input_tensor = input_tensor_intermediate.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze(dim=0)
        top3_prob, top3_idx = torch.topk(probabilities, 3)
    results = []
    for prob, idx in zip(top3_prob, top3_idx):
        breed = class_names[int(idx.item())]
        confidence = prob.item() * 100
        results.append((breed, confidence))
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python test_model_inference.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        sys.exit(1)
    model = load_model()
    predictions = predict_image(model, image_path)
    print(f"Top 3 predictions for {image_path}:")
    for i, (breed, confidence) in enumerate(predictions, 1):
        print(f"{i}. {breed} - {confidence:.2f}%")

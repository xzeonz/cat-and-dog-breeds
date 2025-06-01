from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import io
import pandas as pd
import os
import requests
from typing import cast

app = FastAPI()

# Allow CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration and Initialization ---
MODEL_PATH = "./best_model.pth"
BREED_DESCRIPTIONS_URL = "https://docs.google.com/spreadsheets/d/1v9o-KPbQUnaWb9qK2hZfAHgOLrve4BKx1k30RUIYM3M/export?format=csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 55

model = None
class_names = [
    'Abyssinian', 'Alaskan Malamute', 'American Bobtail', 'American Shorhair', 'American bulldog',
    'American_pit_bull_terrier', 'Basset_hound', 'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer',
    'British_shorthair', 'Bulldog', 'Calico', 'Chihuahua', 'Dachshund', 'Egyptian_mau',
    'English_cocker_paniel', 'English_setter', 'German_Shepherd', 'German_shorthairaired',
    'Golden_Retreiver', 'Great_pyrenees', 'Havanese', 'Husky', 'Japanese chin', 'Keeshond',
    'Labrador_Retriever', 'Leonberger', 'Maine_coon', 'Miniature_pinscher', 'Munchkin',
    'Newfoundland', 'Norwegian_Forest_Cat', 'Ocicat', 'Persian', 'Pomeranian', 'Poodle', 'Pug',
    'Ragdoll', 'Rottweiler', 'Russian_blue', 'Saint_bernard', 'Samoyed', 'Scottish_Fold',
    'Scottish_terrier', 'Shiba_inu', 'Siamese', 'Sphynx', 'Staffordshire_bull_terrier',
    'Tortoiseshell', 'Tuxedo', 'Wheaten_tersier', 'Yorkshire_terrier'
]

breed_descriptions_df = pd.DataFrame()

# Image transformations (must match your training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    global model
    try:
        model = models.efficientnet_b0(pretrained=False)
        num_ftrs: int = model.classifier[1].in_features  # type: ignore
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
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Error loading model: {e}")

def load_breed_descriptions():
    global breed_descriptions_df
    try:
        response = requests.get(BREED_DESCRIPTIONS_URL)
        response.raise_for_status()
        from io import StringIO
        csv_data = StringIO(response.text)
        breed_descriptions_df = pd.read_csv(csv_data)
        print("Loaded breed descriptions with columns:", breed_descriptions_df.columns.tolist())
    except Exception as e:
        print(f"Error loading breed descriptions: {e}")
        breed_descriptions_df = pd.DataFrame()

@app.on_event("startup")
async def startup_event():
    load_model()
    load_breed_descriptions()

def get_breed_description(breed_name: str) -> str:
    if breed_descriptions_df.empty or breed_descriptions_df.shape[1] < 2:
        return "Description not available."

    breed_col = breed_descriptions_df.columns[0]
    row = breed_descriptions_df[
        breed_descriptions_df[breed_col].astype(str).str.strip().str.lower() == breed_name.strip().lower()
    ]

    if not row.empty:
        # Ambil kolom ke 1 sampai ke 6 (maksimum)
        max_index = min(6, breed_descriptions_df.shape[1] - 1)
        return '\n'.join(str(row.iloc[0, i]) for i in range(1, max_index + 1))

    return "Description not available."

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file. Please provide a valid image (e.g., JPG, PNG).")

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please try again later or check server logs.")

    input_tensor = cast(torch.Tensor, transform(image)).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze(dim=0)
        top3_prob, top3_idx = torch.topk(probabilities, 3)

    results = []
    for prob, idx in zip(top3_prob, top3_idx):
        breed = class_names[int(idx.item())]
        description = get_breed_description(breed)
        results.append({
            "breed": breed,
            "confidence": f"{prob.item()*100:.2f}%",
            "description": description
        })

    confidence_message = None
    if top3_prob[0].item() < 0.70:
        confidence_message = "**Pesan: Pastikan gambar jelas! Konfidensi rendah.**"

    return JSONResponse(content={
        "predictions": results,
        "confidence_message": confidence_message
    })

@app.get("/")
async def root():
    return {"message": "Welcome to the Pet Breed Prediction API!"}

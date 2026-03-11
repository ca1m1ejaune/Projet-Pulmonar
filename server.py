from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

CHEXNET_PATH = r"C:\Users\callu\Projet PULMONAR\chexnet.pth.tar"
MAIN_MODEL_PATH = r"C:\Users\callu\Projet PULMONAR\mon_modele_pulmonar_final.pth"
RF_MODEL_PATH = 'rf_siamese_model.pkl'
SCALER_PATH = 'siamese_scaler.pkl'

CLASS_NAMES_MAIN = ['Covid-19', 'Normal', 'Bacterial Pneumonia', 'Viral Pneumonia']
# Based on your siamese.py target_classes order
CLASS_NAMES_SIAMESE = ['Viral Pneumonia', 'Bacterial Pneumonia'] 

device = torch.device('cpu')

# 1. LOAD MAIN DENSENET MODEL
def load_main_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 4)
    model.load_state_dict(torch.load(MAIN_MODEL_PATH, map_location=device))
    model.eval()
    return model

# 2. LOAD SIAMESE FEATURE EXTRACTOR 
def load_feature_extractor():
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    
    # Load CheXNet exactly as in siamese.py
    checkpoint = torch.load(CHEXNET_PATH, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "").replace("densenet121.", "")
        new_k = new_k.replace("norm.1", "norm1").replace("norm.2", "norm2")
        new_k = new_k.replace("conv.1", "conv1").replace("conv.2", "conv2")
        new_k = new_k.replace("classifier.0", "classifier")
        new_state_dict[new_k] = v

    model.classifier = nn.Linear(num_ftrs, 14)
    model.load_state_dict(new_state_dict, strict=False)
    
    # Replace classifier with Identity for feature extraction
    model.classifier = nn.Identity()
    model.eval()
    return model

print("Loading models into memory...")
main_model = load_main_model()
feature_extractor = load_feature_extractor()
rf_classifier = joblib.load(RF_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("All models loaded successfully!")

# HELPER FUNCTIONS
def process_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

def extract_siamese_features(tensor, model):
    with torch.no_grad():
        feat_global = model(tensor)
        
        # Split lungs
        left_lung = tensor[:, :, :, :112]
        right_lung = tensor[:, :, :, 112:]
        right_lung_flipped = torch.flip(right_lung, dims=[3])

        feat_left = model(left_lung)
        feat_right = model(right_lung_flipped)

        # Calculate metrics
        cos_sim = F.cosine_similarity(feat_left, feat_right).unsqueeze(1)
        l1_dist = torch.sum(torch.abs(feat_left - feat_right), dim=1, keepdim=True)
        l2_dist = torch.norm(feat_left - feat_right, p=2, dim=1, keepdim=True)
        chebyshev_dist = torch.max(torch.abs(feat_left - feat_right), dim=1, keepdim=True).values
        dot_product = torch.sum(feat_left * feat_right, dim=1, keepdim=True)

        combined = torch.cat((feat_global, cos_sim, l1_dist, l2_dist, chebyshev_dist, dot_product), dim=1)
        
    return combined.cpu().numpy()

# API ROUTE
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img_bytes = file.read()
    tensor = process_image(img_bytes)

    # STEP 1: Predict with main model
    with torch.no_grad():
        outputs = main_model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        main_confidence, predicted_class = torch.max(probs, 1)
        
    initial_diagnosis = CLASS_NAMES_MAIN[predicted_class.item()]

    # STEP 2: Cascade logic
    if initial_diagnosis in ['Bacterial Pneumonia', 'Viral Pneumonia']:
        # Extract features using the Siamese logic
        features = extract_siamese_features(tensor, feature_extractor)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict with Random Forest
        rf_probs = rf_classifier.predict_proba(features_scaled)[0]
        rf_pred_idx = rf_classifier.predict(features_scaled)[0]
        
        final_diagnosis = CLASS_NAMES_SIAMESE[rf_pred_idx]
        final_confidence = rf_probs[rf_pred_idx] * 100
        
        return jsonify({
            'diagnosis': final_diagnosis,
            'confidence': f"{final_confidence:.2f}%",
            'used_siamese_model': True,
            'initial_model_guess': initial_diagnosis
        })
        
    else:
        # If it's Normal or Covid-19, return the main model's result directly
        return jsonify({
            'diagnosis': initial_diagnosis,
            'confidence': f"{main_confidence.item()*100:.2f}%",
            'used_siamese_model': False
        })

if __name__ == '__main__':
    app.run(port=5000, debug=True)

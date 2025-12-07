from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

CLASS_NAMES = ['Covid-19','Normal' ,'Bacterial Pneumonia' , 'Viral Pneumonia']

def load_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 4)
    # Load weights on CPU
    model.load_state_dict(torch.load(r'C:\Users\callu\Projet PULMONAR\venv\Pullmonar\mon_modele_pulmonar_pytorch.pth', 
                                     map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def process_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img_bytes = file.read()
    tensor = process_image(img_bytes)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

    return jsonify({
        'diagnosis': CLASS_NAMES[predicted_class.item()],
        'confidence': f"{confidence.item()*100:.2f}%"
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)
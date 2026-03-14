import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Importations spécifiques à Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==========================================
# 1. Configuration et Paramètres
# ==========================================
DATA_PATH = r"C:\Users\callu\Projet PULMONAR_Preprocessed"
CHEXNET_PATH = r"C:\Users\callu\Projet PULMONAR\chexnet.pth.tar"
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du matériel : {device}")

# ==========================================
# 2. Préparation des Données et Classes
# ==========================================
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

# Récupération des classes
class_names = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

# ==========================================
# 3. Chargement du Modèle
# ==========================================
print("Chargement et configuration du modèle...")
model = models.densenet121(weights=None)
num_ftrs = model.classifier.in_features

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

try:
    model.load_state_dict(new_state_dict, strict=True)
except RuntimeError:
    model.load_state_dict(new_state_dict, strict=False)

# Remplacement final pour vos 4 classes
model.classifier = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)
model.eval() 

# ==========================================
# 4. Sélection des images spécifiques
# ==========================================
# Dictionnaire associant la classe au nom exact du fichier souhaité
images_cible = {
    "COVID-19": "COVID-19 (1032).jpg",
    "Normal": "Normal (2165).jpg",
    "Pneumonia-Bacterial": "Pneumonia-Bacterial (2174).jpg",
    "Pneumonia-Viral": "Pneumonia-Viral (755).jpg"
}

print("\nChargement des images spécifiques...")
images_to_test = {}

for class_name, img_name in images_cible.items():
    img_path = os.path.join(DATA_PATH, class_name, img_name)
    
    if os.path.exists(img_path):
        print(f"[{class_name}] Image chargée : {img_name}")
        pil_img = Image.open(img_path).convert('RGB')
        img_tensor = data_transforms(pil_img)
        images_to_test[class_name] = (img_tensor, class_to_idx[class_name], img_name)
    else:
        print(f"ATTENTION : L'image {img_path} est introuvable.")

# ==========================================
# 5. Configuration de Grad-CAM
# ==========================================
target_layers = [model.features.norm5]
cam = GradCAM(model=model, target_layers=target_layers)

# ==========================================
# 6. Génération et Affichage Amélioré
# ==========================================
# Agrandissement de la figure pour donner de l'espace (largeur 12, hauteur proportionnelle)
fig, axes = plt.subplots(len(images_to_test), 2, figsize=(14, 6 * len(images_to_test)))

# Titre principal plus grand et espacé
fig.suptitle("Analyse Grad-CAM des pathologies pulmonaires", fontsize=20, fontweight='bold', y=0.98)

def denormalize(tensor):
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    tensor = inv_normalize(tensor)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.permute(1, 2, 0).numpy()

# Sécurité si une seule image est trouvée (axes devient 1D)
if len(images_to_test) == 1:
    axes = np.expand_dims(axes, axis=0)

for idx, (class_name, data) in enumerate(images_to_test.items()):
    img_tensor, label_idx, file_name = data
    
    input_tensor = img_tensor.unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(label_idx)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    rgb_img = denormalize(img_tensor)
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Paramétrage de la colonne 1 (Image Originale)
    axes[idx, 0].imshow(rgb_img)
    axes[idx, 0].set_title(f"Classe : {class_name}\nFichier : {file_name}", fontsize=14, pad=12)
    axes[idx, 0].axis('off')
    
    # Paramétrage de la colonne 2 (Grad-CAM)
    axes[idx, 1].imshow(cam_image)
    axes[idx, 1].set_title(f"Grad-CAM (Activation)", fontsize=14, pad=12)
    axes[idx, 1].axis('off')

# L'espacement magique pour éviter que les titres et les images se chevauchent
plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=4.0, w_pad=2.0)
plt.show()
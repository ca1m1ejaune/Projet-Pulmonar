import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import LabelBinarizer
import copy
import os

# Config des paramètres
DATA_PATH = r"C:\Users\callu\Projet PULMONAR_Preprocessed"
CHEXNET_PATH = r"C:\Users\callu\Projet PULMONAR\chexnet.pth.tar"
BATCH_SIZE = 32
IMG_SIZE = 224

# Hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du matériel : {device}")

# Préparation des données  
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

full_dataset_raw = datasets.ImageFolder(DATA_PATH, transform=data_transforms)

target_classes = ['Pneumonia-Viral', 'Pneumonia-Bacterial']
class_to_idx = full_dataset_raw.class_to_idx

target_indices_in_original = [class_to_idx[cls] for cls in target_classes] 

mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(target_indices_in_original)}

filtered_samples = []

for path, label in full_dataset_raw.samples:
    if label in target_indices_in_original:
        filtered_samples.append((path, mapping[label]))

full_dataset_raw.samples = filtered_samples
full_dataset_raw.imgs = filtered_samples 
full_dataset_raw.classes = target_classes
full_dataset_raw.class_to_idx = {cls: i for i, cls in enumerate(target_classes)}
 
train_size = int(0.8 * len(full_dataset_raw))
val_size = len(full_dataset_raw) - train_size
train_dataset, val_dataset = random_split(full_dataset_raw, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Nouveaux indices: {full_dataset_raw.class_to_idx}")

# Création du modèle

print("Chargement et configuration du modèle...")
model = models.densenet121(weights=None)
num_ftrs = model.classifier.in_features

# regarder d'autres archis plus petites pour mettre dans le clf

# Chargement des poids CheXNet 
checkpoint = torch.load(CHEXNET_PATH, map_location=device)
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

new_state_dict = {}
for k, v in state_dict.items():
    new_k = k.replace("module.", "").replace("densenet121.", "")
    new_k = new_k.replace("norm.1", "norm1").replace("norm.2", "norm2")
    new_k = new_k.replace("conv.1", "conv1").replace("conv.2", "conv2")
    new_k = new_k.replace("classifier.0", "classifier")
    new_state_dict[new_k] = v

model.classifier = nn.Linear(num_ftrs, 14) # Temporaire pour charger les poids

try:
    model.load_state_dict(new_state_dict, strict=True)
    print("Poids CheXnet chargés (Strict Mode).")
except RuntimeError as e:
    print(f"Chargement Strict échoué, passage en mode souple... Erreur: {e}")
    model.load_state_dict(new_state_dict, strict=False)

# Pas de couche finale 
model.classifier = nn.Identity()

for param in model.parameters():
    param.requires_grad = False

model = model.to(device)
model.eval()

import torch.nn.functional as F

# MODIFICATION : Extraction features optimisée
def extract_siamese_features_OPTIMIZED(loader, model):
    features_list = []
    labels_list = []

    print("Extraction des features (version optimisée)...")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)

            # Features globales
            feat_global = model(inputs)

            # Métriques siamois
            left_lung = inputs[:,:,:,:112]
            right_lung = inputs[:,:,:,112:]
            right_lung_flipped = torch.flip(right_lung, dims=[3])

            feat_left = model(left_lung)
            feat_right = model(right_lung_flipped)

            cos_sim = F.cosine_similarity(feat_left, feat_right).unsqueeze(1)
            l1_dist = torch.sum(torch.abs(feat_left - feat_right), dim=1, keepdim=True)
            l2_dist = torch.norm(feat_left - feat_right, p=2, dim=1, keepdim=True)
            chebyshev_dist = torch.max(torch.abs(feat_left - feat_right), dim=1, keepdim=True).values # Distance L-infini
            dot_product = torch.sum(feat_left * feat_right, dim=1, keepdim=True) # Produit scalaire

            # mohri, regarde d'autres distances. 

            combined = torch.cat((feat_global, cos_sim, l1_dist, l2_dist, chebyshev_dist, dot_product), dim=1)

            features_list.append(combined.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    return np.vstack(features_list), np.concatenate(labels_list)

# MODIFICATION 2 Extraction
X_train, y_train = extract_siamese_features_OPTIMIZED(train_loader, model)
X_val, y_val = extract_siamese_features_OPTIMIZED(val_loader, model)

# Diagnostics
print(f"\n Diagnostics :")
print(f"Shape : {X_train.shape}")
print(f"Similarité cosinus (Viral) : {X_train[y_train==0, -5].mean():.4f}")
print(f"Similarité cosinus (Bacterial) : {X_train[y_train==1, -5].mean():.4f}")
print(f"Distance L1 (Viral) : {X_train[y_train==0, -4].mean():.4f}")
print(f"Distance L1 (Bacterial) : {X_train[y_train==1, -4].mean():.4f}")

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import optuna

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# MODIFICATION : Optuna avec F1-macro + class_weight
from sklearn.metrics import f1_score

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 15)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    
    clf_opt = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        class_weight='balanced',  # <-- Important pour le déséquilibre des classes 
        random_state=42,
        n_jobs=-1
    )
    
    clf_opt.fit(X_train_scaled, y_train)
    y_pred = clf_opt.predict(X_val_scaled)
    
    return f1_score(y_val, y_pred, average='macro')  

print("Optimisation bayésienne...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print(f"Meilleurs paramètres : {study.best_params}")
print(f"Meilleur F1-macro : {study.best_value:.4f}")

# MODIFICATION : Modèle final avec class_weight 
best_params = study.best_params
clf = RandomForestClassifier(
    **best_params, 
    class_weight='balanced',  
    random_state=42, 
    n_jobs=-1
)
clf.fit(X_train_scaled, y_train)
# Feature importance
importances = clf.feature_importances_

# Mise à jour avec les 5 distances exactes dans l'ordre du torch.cat
feature_names = [f"DenseNet_{i}" for i in range(1024)] + ["Cosine_Sim", "L1_Dist", "L2_Dist", "Chebyshev_Dist", "Dot_Product"]

print(f"\nTop 15 features importantes :")
# J'ai monté à 15 pour qu'on ait plus de chances de voir si mes distances ont un fort impact, mais finalement on voit que des features DenseNet.
top_indices = np.argsort(importances)[-15:] 
for idx in reversed(top_indices):
    print(f"  {feature_names[idx]}: {importances[idx]:.4f}")

# MODIFICATION : Seuil optimal 
from sklearn.metrics import precision_recall_curve

y_train_probs = clf.predict_proba(X_train_scaled)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\nSeuil optimal : {optimal_threshold:.3f} (défaut = 0.50)")

# Prédictions avec seuil par défaut
val_preds_default = clf.predict(X_val_scaled)
val_probs = clf.predict_proba(X_val_scaled)[:, 1]

# Prédictions avec seuil optimal
val_preds_optimal = (val_probs >= optimal_threshold).astype(int)

# Résultats
print("\n" + "="*60)
print("RÉSULTATS AVEC SEUIL PAR DÉFAUT (0.5)")
print("="*60)
print(classification_report(y_val, val_preds_default, target_names=target_classes))

print("\n" + "="*60)
print(f"RÉSULTATS AVEC SEUIL OPTIMAL ({optimal_threshold:.3f})")
print("="*60)
print(classification_report(y_val, val_preds_optimal, target_names=target_classes))

# Feature importance
feature_names = [f"DenseNet_{i}" for i in range(1024)] + ["Cosine_Sim", "L1_Dist", "L2_Dist", "Chebyshev_Dist", "Dot_Product"]
importances = clf.feature_importances_

print(f"\nTop 10 features importantes :")
top_indices = np.argsort(importances)[-10:]
for idx in reversed(top_indices):
    print(f"  {feature_names[idx]}: {importances[idx]:.4f}")

# Courbe ROC
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_val, val_probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Matrice de confusion avec seuil optimal
from sklearn.metrics import ConfusionMatrixDisplay

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ConfusionMatrixDisplay.from_predictions(y_val, val_preds_default, 
                                        display_labels=target_classes, ax=ax1)
ax1.set_title(f'Seuil par défaut (0.5)')

ConfusionMatrixDisplay.from_predictions(y_val, val_preds_optimal, 
                                        display_labels=target_classes, ax=ax2)
ax2.set_title(f'Seuil optimal ({optimal_threshold:.3f})')

plt.tight_layout()
plt.show()

# Sauvegarde du modèle finale

import joblib

# Sauvegarder le modèle ET le scaler !
joblib.dump(clf, 'rf_siamese_model.pkl')
joblib.dump(scaler, 'siamese_scaler.pkl')
print("Modèles sauvegardés pour le backend Flask !")

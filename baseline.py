import os, math, random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import (classification_report, 
                             confusion_matrix,roc_curve, 
                             auc)  # métriques

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from itertools import cycle

DATA_DIR = r"C:\Users\callu\Projet PULMONAR"  
BATCH_SIZE = 32
LR = 1e-5
EPOCHS = 20            # nombre d'époques d'entraînement
SEED = 42              # seed pour la reproductibilité
IMG_SIZE = 224         # taille des images d'entrée (ResNet)

# On fixe les seeds pour rendre les résultats reproductibles
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# On choisit le device (GPU si dispo, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

NUM_WORKERS = 0 if os.name == "nt" else 4   # nombres de workers pour le DataLoader
PIN_MEM = device.type == "cuda"             # pin_memory activé si GPU

# Transformations pour l'entraînement (avec data augmentation)
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Transformations pour la validation / test (pas de data augmentation)
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# On charge une première fois le dataset juste pour gérer le split
full_dataset_for_split = datasets.ImageFolder(DATA_DIR)  
classes = full_dataset_for_split.classes
print("Classes:", classes)

# On récupère les labels et on regroupe les indices par classe
targets = full_dataset_for_split.targets  
indices_by_class = defaultdict(list)
for idx, t in enumerate(targets):
    indices_by_class[t].append(idx)

# On fait un split stratifé 70/15/15 à la main
train_idx, val_idx, test_idx = [], [], []
for c, idxs in indices_by_class.items():
    n = len(idxs)
    n_train = int(round(0.70 * n))
    n_val   = int(round(0.15 * n))
    random.shuffle(idxs)
    train_idx += idxs[:n_train]
    val_idx   += idxs[n_train:n_train+n_val]
    test_idx  += idxs[n_train+n_val:]

# Vérifs rapides
def pct(n, d): return f"{100*n/d:.1f}%"
print(f"Split sizes -> train: {len(train_idx)} ({pct(len(train_idx), len(targets))}), "
      f"val: {len(val_idx)} ({pct(len(val_idx), len(targets))}), "
      f"test: {len(test_idx)} ({pct(len(test_idx), len(targets))})")

# On recharge le dataset avec les bonnes transforms
train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)
val_dataset   = datasets.ImageFolder(DATA_DIR, transform=eval_tf)
test_dataset  = datasets.ImageFolder(DATA_DIR, transform=eval_tf)

# On applique les indices de split avec Subset
train_dataset = Subset(train_dataset, train_idx) 
val_dataset   = Subset(val_dataset,   val_idx)
test_dataset  = Subset(test_dataset,  test_idx)

# DataLoaders pour itérer sur les batches
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEM)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEM)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEM)

#pin memory alloue la mémoire

# Chargement du modèle pré-entraîné sur ImageNet
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

#Rajout des blocs de fine tuning/teste resnet50 

for param in model.parameters():
    param.requires_grad = True
#Là on a juste rajouter un bloc pour geler les poids et que l'optimiseur ne voie que la dernière couche 

# On remplace la dernière couche fully-connected pour 4 classes
in_f = model.fc.in_features
model.fc = nn.Linear(in_f, 4) 
model = model.to(device)

# Fonction de coût et optimiseur
criterion = nn.CrossEntropyLoss() #peut-être vérifier la fonction de coût
optimizer = torch.optim.Adam(model.parameters(), lr=LR) #là on enlève .fc pour que Resnet puisse être fine-tuné entièrement

def run_epoch(loader, train: bool):
    """
    Boucle d'entraînement / validation pour une époque.
    Retourne la loss moyenne et l'accuracy.
    """
    if train:
        model.train()
    else:
        model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad()

        # On active / désactive les gradients selon le mode
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

        # Accumulation de la loss et de l'accuracy
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(1)
        total_correct += (preds == labels).sum().item()
        total_count += images.size(0)

    avg_loss = total_loss / total_count
    acc = total_correct / total_count
    return avg_loss, acc

best_val_acc = 0.0

# Liste pour stocker l'historique des métriques 
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

# Boucle principale d'entraînement
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader, train=False)
    
    # stocke les métriques
    train_loss_history.append(tr_loss)
    train_acc_history.append(tr_acc)
    val_loss_history.append(va_loss)
    val_acc_history.append(va_acc)

    print(f"[Epoch {epoch:02d}/{EPOCHS}] "
          f"train_loss={tr_loss:.4f} train_acc={tr_acc*100:.2f}% | "
          f"val_loss={va_loss:.4f} val_acc={va_acc*100:.2f}%")

# Évaluation sur le test set + collecte pour les métriques
test_loss, test_correct, test_count = 0.0, 0, 0
all_preds = []
all_labels = []
all_probs = []  # Pour stocker les probabilités pour ROC

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        # Accumulation de la loss et de l'accuracy
        test_loss += loss.item() * images.size(0)
        preds = logits.argmax(1)
        test_correct += (preds == labels).sum().item()
        test_count += images.size(0)
        
        # Calculate probabilities (Softmax) for ROC
        probs = torch.softmax(logits, dim=1)

        # On stocke pour la matrice de confusion / rapport
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist()) # Store probabilities

# Moyenne de la loss et accuracy globale test
test_loss /= test_count
test_acc = test_correct / test_count
# Moyenne de la loss et accuracy globale test
test_loss /= test_count
test_acc = test_correct / test_count

print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc*100:.2f}%")

# Tracé des courbes ROC pour chaque classe
# Rapport de classification (précision, recall, f1-score par classe)
print("\nClassification report:")
print(classification_report(all_labels, all_preds, target_names=classes))

# Matrice de confusion
print("Confusion matrix:")
cm = confusion_matrix(all_labels, all_preds)
print(cm)

#Tracé des courbes d'entraînement et de validation
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), train_loss_history, label='Train Loss')
plt.plot(range(1, EPOCHS + 1), val_loss_history, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), train_acc_history, label='Train Acc')
plt.plot(range(1, EPOCHS + 1), val_acc_history, label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#tracé des courbes ROC pour chaque classe
y_test_bin = label_binarize(all_labels, classes=[0, 1, 2, 3])
y_score = np.array(all_probs)
n_classes = y_test_bin.shape[1]

# Calcul des courbes ROC et AUC pour chaque classe
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Tracé des courbes ROC
plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'red', 'green', 'orange'])

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2) 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

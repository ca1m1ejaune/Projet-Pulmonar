import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import os
import re

# Configuration des chemins et parametres
DATA_PATH = r"C:\Users\callu\Projet PULMONAR_Preprocessed"
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 20

# Verification si j'utilise le GPU ou le CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du matériel : {device}")
if device.type == 'cuda':
    print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")

# PREPARATION DES DONNEES

# On prepare les transformations pour les images
# C'est important de normaliser comme ImageNet pour que le transfert learning marche bien
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

# Chargement des images depuis le dossier
full_dataset = datasets.ImageFolder(DATA_PATH, transform=data_transforms)

# On decoupe en entrainement (80%) et validation (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Les dataloaders permettent de charger les données par petits paquets
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = full_dataset.classes
print(f"Classes détectées : {class_names}")
print(f"Images entrainement : {len(train_dataset)}, Images validation : {len(val_dataset)}")


# CREATION DU MODELE (DENSENET121 + CHEXNET)

print("Configuration du modèle avec poids CheXnet...")
# Chemin vers les poids pre-entraines telecharges
CHEXNET_PATH = r"C:\Users\callu\Projet PULMONAR\chexnet.pth.tar"

# On charge l'architecture vide de base
model = models.densenet121(weights=None)
num_ftrs = model.classifier.in_features

# Chargement du fichier de poids brut
checkpoint = torch.load(CHEXNET_PATH, map_location=device)

# Petit fix au cas ou c'est un dictionnaire ou directement les poids
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# NETTOYAGE DES CLES DU DICTIONNAIRE
# C'est la partie un peu technique pour que les noms des couches correspondent
new_state_dict = {}
for k, v in state_dict.items():
    new_k = k
    
    # On retire les prefixes bizarres qui font planter le chargement
    new_k = new_k.replace("module.", "")
    new_k = new_k.replace("densenet121.", "")
    
    # Correction de syntaxe specifique a ce modele  
    new_k = new_k.replace("norm.1", "norm1")
    new_k = new_k.replace("norm.2", "norm2")
    new_k = new_k.replace("conv.1", "conv1")
    new_k = new_k.replace("conv.2", "conv2")
    
    # On renomme la couche finale pour qu'elle matche
    new_k = new_k.replace("classifier.0", "classifier")

    new_state_dict[new_k] = v

# Injection des poids nettoyes
# On met temporairement 14 classes car CheXnet a ete entraine sur 14 maladies
model.classifier = nn.Linear(num_ftrs, 14)

try:
    print("Tentative de chargement des poids nettoyés...")
    model.load_state_dict(new_state_dict, strict=True)
    print("Poids CheXnet chargés avec succès !")
except RuntimeError as e:
    print(f"Petite erreur de chargement, on réessaie en mode souple : \n{e}")
    model.load_state_dict(new_state_dict, strict=False)

# Je remplace la derniere couche pour mes 4 classes a moi
model.classifier = nn.Linear(num_ftrs, 4)

# On envoie tout sur la carte graphique
model = model.to(device)

# Definition de la fonction de cout et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 3. BOUCLE D'ENTRAINEMENT

# Listes pour sauvegarder les resultats et faire les graphiques apres
train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []

print("C'est parti pour l'entrainement...")

for epoch in range(EPOCHS):
    # Mode entrainement active
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # On remet les gradients a zero avant de commencer
        optimizer.zero_grad()

        # Passage dans le modele
        outputs = model(inputs) 
        loss = criterion(outputs, labels)
        
        # Calcul des erreurs et mise a jour des poids
        loss.backward()
        optimizer.step()

        # Calcul de la precision
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    
    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc.item())

    # Mode validation active (on ne touche pas aux poids)
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    
    with torch.no_grad(): # Pas besoin de garder les gradients en memoire ici
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)

    val_loss = val_running_loss / len(val_dataset)
    val_acc = val_running_corrects.double() / len(val_dataset)
    
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc.item())

    print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

print("Fini ! Le modele est entrainé.")

# ANALYSE DES RESULTATS

print("Creation des graphiques...")

# Affichage des courbes de Loss et Accuracy
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Preparation des donnees pour la matrice de confusion et ROC
all_preds = []
all_labels = []
all_probs = []

model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        
        # On recupere les probabilites pour la courbe ROC
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Affichage de la Matrice de Confusion
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Classe Predite')
plt.ylabel('Vraie Classe')
plt.title('Matrice de Confusion')
plt.show()

# Affichage des Courbes ROC
lb = LabelBinarizer()
lb.fit(all_labels)
y_true_bin = lb.transform(all_labels)

plt.figure(figsize=(10, 8))
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC {class_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Faux Positifs')
plt.ylabel('Vrais Positifs')
plt.title('Courbes ROC par classe')
plt.legend(loc="lower right")
plt.show()

# Sauvegarde du fichier final
torch.save(model.state_dict(), 'mon_modele_pulmonar_pytorch.pth')
print("Sauvegarde effectuee sous 'mon_modele_pulmonar_pytorch.pth'")
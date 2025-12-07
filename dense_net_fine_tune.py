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

# CLASS EARLY STOPPING 
class EarlyStopping:
    """
    Arrête l'entraînement si la loss de validation ne s'améliore pas après un certain nombre d'epochs.
    """
    def __init__(self, patience=5, min_delta=0, path='checkpoint_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Sauvegarde le modèle quand la loss de validation diminue.'''
        self.best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), self.path)
        print(f'Validation loss decreased. Saving model...')

# CONFIGURATION 
DATA_PATH = r"C:\Users\callu\Projet PULMONAR_Preprocessed"
CHEXNET_PATH = r"C:\Users\callu\Projet PULMONAR\chexnet.pth.tar"
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 100
PATIENCE = 5         # Nombre d'epochs à attendre avant Early Stopping
FINE_TUNE = True      # True = Entraîner tout le modèle, False = Entraîner seulement le classifieur

# Hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du matériel : {device}")

# PREPARATION DES DONNEES 
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

full_dataset = datasets.ImageFolder(DATA_PATH, transform=data_transforms)

# Split 80/20
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = full_dataset.classes
print(f"Classes : {class_names}")

# --- CREATION DU MODELE ---

print("Chargement et configuration du modèle...")
model = models.densenet121(weights=None)
num_ftrs = model.classifier.in_features

# Chargement des poids CheXNet (Code existant)
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

# Remplacement de la couche finale pour nos 4 classes
model.classifier = nn.Linear(num_ftrs, 4)

# BLOC FINE TUNING 
if FINE_TUNE:
    print("Mode FINE-TUNING activé : Tous les paramètres sont entraînables.")
    # C'est ici le code que vous vouliez ajouter :
    for param in model.parameters():
        param.requires_grad = True
else:
    print("Mode FEATURE EXTRACTION : Seule la dernière couche est entraînable.")
    # On gèle tout d'abord
    for param in model.parameters():
        param.requires_grad = False
    # On dégèle uniquement le classifieur
    for param in model.classifier.parameters():
        param.requires_grad = True

model = model.to(device)

# OPTIMIZER & LOSS
criterion = nn.CrossEntropyLoss()
# On optimise uniquement les paramètres qui ont requires_grad=True
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)

# Initialisation Early Stopping
early_stopping = EarlyStopping(patience=PATIENCE, path='checkpoint_temp.pth')

# BOUCLE D'ENTRAINEMENT 
train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []

print(f"Démarrage de l'entraînement pour {EPOCHS} epochs max...")

for epoch in range(EPOCHS):
    # 1. Train
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs) 
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    
    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc.item())

    # 2. Validation
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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

    # 3. Check Early Stopping
    early_stopping(val_loss, model)
    
    if early_stopping.early_stop:
        print("Early stopping déclenché !")
        break

# IMPORTANT: Recharger les meilleurs poids enregistrés par l'Early Stopping
print("Chargement des meilleurs poids retenus par l'Early Stopping...")
model.load_state_dict(torch.load('checkpoint_temp.pth'))

# ANALYSE DES RESULTATS 
print("Génération des graphiques...")

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(train_acc_history, label='Train Acc')
plt.plot(val_acc_history, label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()

# Matrice de Confusion et ROC
all_preds = []
all_labels = []
all_probs = []

model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# --- NOUVEAU: RAPPORT DE CLASSIFICATION ---
print("\n--- Rapport de Classification ---")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Heatmap
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Matrice de Confusion')
plt.show()

# ROC
lb = LabelBinarizer()
lb.fit(all_labels)
y_true_bin = lb.transform(all_labels)

plt.figure(figsize=(10, 8))
for i in range(len(class_names)):
    # Gestion du cas binaire vs multi-classe pour ROC
    if y_true_bin.shape[1] == 1: 
        # Cas binaire (0 ou 1)
        target = y_true_bin[:, 0]
        score = all_probs[:, 1] # Proba de la classe positive
        fpr, tpr, _ = roc_curve(target, score)
    else:
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC {class_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()

# Sauvegarde finale
torch.save(model.state_dict(), 'mon_modele_pulmonar_final.pth')
print("Sauvegarde terminée.")
# -----------------------------
# 1️⃣ Importer les bibliothèques
# -----------------------------
import os
import numpy as np
import librosa

# -----------------------------
# 2️⃣ Paramètres
# -----------------------------
DATASET_PATH = "/content/drive/MyDrive/processed_dataset"  # chemin dataset processed
TARGET_SR = 16000
TARGET_LEN = 16000
N_MFCC = 30  # meilleure valeur choisie

# -----------------------------
# 3️⃣ Initialiser X et y
# -----------------------------
X = []
y = []
labels = []

# -----------------------------
# 4️⃣ Parcourir les dossiers et extraire MFCC
# -----------------------------
for idx, word in enumerate(os.listdir(DATASET_PATH)):
    labels.append(word)
    folder_path = os.path.join(DATASET_PATH, word)
    
    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            
            # Charger audio
            audio, sr = librosa.load(file_path, sr=TARGET_SR)
            
            # Extraire MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
            
            # Normaliser MFCC
            mfcc = (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc))
            
            # Ajouter dimension channel pour CNN (si besoin)
            mfcc = mfcc[..., np.newaxis]  # shape = (N_MFCC, frames, 1)
            
            X.append(mfcc)
            y.append(idx)

# Convertir en numpy array
X = np.array(X)
y = np.array(y)

print("✅ Extraction MFCC terminée")
print("X shape:", X.shape)
print("y shape:", y.shape)

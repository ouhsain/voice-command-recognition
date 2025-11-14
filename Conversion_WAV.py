# -----------------------------
# 1️⃣ Importer les bibliothèques
# -----------------------------
import os
import numpy as np
import librosa
import soundfile as sf

# -----------------------------
# 2️⃣ Définir les chemins
# -----------------------------
# Remplace ces chemins par le chemin exact vers ton dataset sur ton PC
INPUT_ROOT = "C:\Users\bakar\OneDrive\Desktop\projet_DL\dataset"
OUTPUT_ROOT = "C:\Users\bakar\OneDrive\Desktop\projet_DL\processed_dataset"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# -----------------------------
# 3️⃣ Paramètres audio
# -----------------------------
TARGET_SR = 16000      # 16 kHz
TARGET_LEN = 16000     # 1 seconde

# -----------------------------
# 4️⃣ Vérifier le dataset
# -----------------------------
if not os.path.exists(INPUT_ROOT):
    raise FileNotFoundError(f"Le dossier dataset n'existe pas à ce chemin : {INPUT_ROOT}")

print("✅ Dataset trouvé. Sous-dossiers :", os.listdir(INPUT_ROOT))

# -----------------------------
# 5️⃣ Conversion des fichiers
# -----------------------------
for word in os.listdir(INPUT_ROOT):
    input_folder = os.path.join(INPUT_ROOT, word)
    
    # Ignorer tout ce qui n'est pas un dossier
    if not os.path.isdir(input_folder):
        continue

    output_folder = os.path.join(OUTPUT_ROOT, word)
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.lower().endswith((".mp3", ".wav", ".ogg", ".flac")):
            in_path = os.path.join(input_folder, file)

            # Charger audio → mono, 16 kHz
            y, sr = librosa.load(in_path, sr=TARGET_SR, mono=True)

            # Supprimer silence début/fin
            y, _ = librosa.effects.trim(y, top_db=25)

            # Normaliser
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))

            # Pad ou trim pour durée fixe
            if len(y) < TARGET_LEN:
                y = np.pad(y, (0, TARGET_LEN - len(y)))
            else:
                y = y[:TARGET_LEN]

            # Sauvegarder en WAV
            out_path = os.path.join(output_folder, file.split('.')[0] + ".wav")
            sf.write(out_path, y, TARGET_SR)

            print("✔ Traité :", out_path)

print("\n🎉 Tous les fichiers audio ont été traités avec succès !")

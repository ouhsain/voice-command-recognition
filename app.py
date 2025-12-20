# voice_app_http.py

import os
import time
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
import requests

# ====== ESP HTTP CONFIG ======
ESP_IP = "10.169.83.183" \
""  # ⚠️ bdelha b IP li banch f Serial Monitor
ESP_BASE_URL = f"http://{ESP_IP}"

# ==============================
# 1️⃣ FEATURE EXTRACTION (same)
# ==============================

def extract_features(file_path, sr=16000, n_mfcc=40, max_sec=2, hop_length=256, n_fft=512):
    audio, _ = librosa.load(file_path, sr=sr)
    audio, _ = librosa.effects.trim(audio, top_db=20)

    max_len = sr * max_sec
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                hop_length=hop_length, n_fft=n_fft)

    target_T = 124
    if mfcc.shape[1] < target_T:
        pad_width = target_T - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :target_T]

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, delta, delta2])
    return combined[..., np.newaxis]  # (120, 124, 1)

# ==============================
# 2️⃣ MODEL LOADING
# ==============================

MODEL_PATH = r"voice_model_darija.keras"  # bdel path ila khass
print("[PC] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[PC] Model loaded.")

labels = ["ch3al", "sini bchwiya", "sini bzarba", "tfi"]

# ==============================
# 3️⃣ RECORD FROM MIC
# ==============================

TEMP_WAV = "temp.wav"

def record_to_wav(path: str, duration: float = 2.0, sr: int = 16000):
    print(f"[PC] Recording {duration}s ...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    sf.write(path, audio, sr)
    print(f"[PC] Saved to {path}")

# ==============================
# 4️⃣ PREDICTION: mic -> label
# ==============================

def predict_label_from_mic() -> str:
    record_to_wav(TEMP_WAV, duration=2.0, sr=16000)

    feats = extract_features(TEMP_WAV)
    feats = np.expand_dims(feats, axis=0)

    preds = model.predict(feats)
    idx = int(np.argmax(preds[0]))
    label = labels[idx]
    conf = float(np.max(preds[0]))
    print(f"[PC] Predicted: {label} (conf = {conf:.3f})")

    try:
        os.remove(TEMP_WAV)
    except:
        pass

    return label

# ==============================
# 5️⃣ SEND HTTP TO ESP
# ==============================

def send_http_command_from_label(label: str):
    # we map label -> same strings used in /cmd?c=...
    label = label.strip().lower()
    if label not in ["ch3al", "tfi", "sini bzarba", "sini bchwiya"]:
        print("[PC] Unknown label for ESP:", label)
        return

    params = {"c": label.replace(" ", "_")}  # "sini bzarba" -> "sini_bzarba"
    url = ESP_BASE_URL + "/cmd"
    try:
        r = requests.get(url, params=params, timeout=2)
        print(f"[PC] HTTP {r.status_code} -> {r.text}")
    except Exception as e:
        print("[PC] HTTP error:", e)

# ==============================
# 6️⃣ MAIN LOOP
# ==============================

def main():
    try:
        while True:
            print("\nPress ENTER باش trecordi (Ctrl+C باش tkhrej)")
            input()
            label = predict_label_from_mic()
            send_http_command_from_label(label)
    except KeyboardInterrupt:
        print("\n[PC] Stopping.")


if __name__ == "__main__":
    main()


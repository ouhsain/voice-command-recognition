from flask import Flask, request, jsonify
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import requests  # to communicate with ESP32

app = Flask(__name__)
model = load_model("voice_model.h5")
labels = ["sini bzarba", "ch3al", "tfi", "sini bchwiya"]

ESP32_URL = "http://192.168.1.50/command"   # change to your ESP32 IP

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc.reshape(1, 1, 40)

@app.route("/predict", methods=["POST"])
def predict():
    audio = request.files["audio"]
    audio_path = "temp.wav"
    audio.save(audio_path)

    features = extract_features(audio_path)
    pred = model.predict(features)[0]
    label = labels[np.argmax(pred)]

    # send command to ESP32
    requests.post(ESP32_URL, json={"command": label})

    return jsonify({"prediction": label})

import os
import json
import random

from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ============================================================
# 1. SETUP FLASK
# ============================================================
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")

CONFIDENCE_THRESHOLD = 0.25
FALLBACK_RESPONSE = (
    "Maaf, saya belum memahami pertanyaan itu. "
    "Coba tanyakan tentang jam buka, cara pinjam buku, atau syarat pendaftaran anggota ya."
)

# ============================================================
# 2. LOAD DATASET INTENT
# ============================================================
with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents_data = json.load(f)

# ============================================================
# 3. TRAINING MODEL (otomatis saat app.py dijalankan)
# ============================================================
def train_model():
    texts, labels = [], []
    responses = {}

    for intent in intents_data["intents"]:
        tag = intent["tag"]
        responses[tag] = intent["responses"]
        for pattern in intent["patterns"]:
            texts.append(pattern.lower())
            labels.append(tag)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, C=10)),
    ])
    pipeline.fit(texts, labels)

    return pipeline, responses


model, intent_responses = train_model()
print(f"✅ Model chatbot berhasil dilatih ({len(intents_data['intents'])} intent)")


# ============================================================
# 4. FUNGSI PREDIKSI
# ============================================================
def get_bot_response(user_message: str) -> str:
    cleaned = user_message.lower().strip()

    proba = model.predict_proba([cleaned])[0]
    classes = model.classes_
    best_idx = proba.argmax()
    best_tag = classes[best_idx]
    best_score = proba[best_idx]

    if best_score < CONFIDENCE_THRESHOLD:
        return FALLBACK_RESPONSE

    return random.choice(intent_responses[best_tag])


# ============================================================
# 5. ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Pesan tidak boleh kosong"}), 400

    reply = get_bot_response(user_message)
    return jsonify({"reply": reply})


# ============================================================
# 6. JALANKAN SERVER
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)
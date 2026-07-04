import os
import json
import random

from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ============================================================
# 1. SETUP FLASK
# ============================================================
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")
LOG_PATH = os.path.join(BASE_DIR, "unanswered_log.txt")

CONFIDENCE_THRESHOLD = 0.25
FALLBACK_RESPONSE = (
    "Maaf, saya belum memahami pertanyaan itu. "
    "Coba tanyakan tentang jam buka, cara pinjam buku, atau daftar kategori buku ya."
)

# ============================================================
# 2. SETUP STEMMER (Bahasa Indonesia)
# ============================================================
stemmer = StemmerFactory().create_stemmer()

def clean_text(text: str) -> str:
    """Bersihkan & stem teks: 'meminjam' -> 'pinjam'."""
    text = text.lower().strip()
    text = stemmer.stem(text)
    return text

# ============================================================
# 3. LOAD DATASET INTENT
# ============================================================
with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents_data = json.load(f)

# ============================================================
# 4. TRAINING MODEL + EVALUASI AKURASI
# ============================================================
def train_model():
    texts, labels = [], []
    responses = {}
    options_map = {}

    for intent in intents_data["intents"]:
        tag = intent["tag"]
        responses[tag] = intent["responses"]
        options_map[tag] = intent.get("options", [])
        for pattern in intent["patterns"]:
            texts.append(clean_text(pattern))
            labels.append(tag)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, C=10)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    akurasi = accuracy_score(y_test, y_pred)
    print(f"📊 Akurasi model (data uji): {akurasi * 100:.2f}%")

    # Latih ulang dengan seluruh data untuk model final
    pipeline.fit(texts, labels)

    return pipeline, responses, options_map


model, intent_responses, intent_options = train_model()
print(f"✅ Model chatbot berhasil dilatih ({len(intents_data['intents'])} intent)")


# ============================================================
# 5. FUNGSI PREDIKSI + LOGGING PERTANYAAN GAGAL
# ============================================================
def get_bot_response(user_message: str):
    cleaned = clean_text(user_message)

    proba = model.predict_proba([cleaned])[0]
    classes = model.classes_
    best_idx = proba.argmax()
    best_tag = classes[best_idx]
    best_score = proba[best_idx]

    if best_score < CONFIDENCE_THRESHOLD:
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{user_message}\n")
        except Exception:
            pass
        return FALLBACK_RESPONSE, []

    reply = random.choice(intent_responses[best_tag])
    options = intent_options.get(best_tag, [])
    return reply, options


# ============================================================
# 6. ROUTES
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

    reply, options = get_bot_response(user_message)
    return jsonify({"reply": reply, "options": options})


# ============================================================
# 7. JALANKAN SERVER
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and vectorizer from local path
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
ensemble_data = joblib.load("model/best_ensemble_model.pkl")
svm_model = ensemble_data['svm']
logreg_model = ensemble_data['logreg']
weights = ensemble_data['weights']

label_map = {0: "Non-Toxic", 1: "Toxic", 2: "sarcasm-positive", 3: "sarcasm-negative", 4: "spam", 5: "Not-Tamil"}

def clean_text(text):
    import re, emoji
    text = text.lower()
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\u0B80-\u0BFF\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_text(text):
    text = clean_text(text)
    features = vectorizer.transform([text])
    svm_probs = svm_model.predict_proba(features)[0]
    logreg_probs = logreg_model.predict_proba(features)[0]
    combined_probs = svm_probs * weights[0] + logreg_probs * weights[1]
    label_num = np.argmax(combined_probs)
    return label_map.get(label_num, "Unknown")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400
        result = predict_text(text)
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/")
def index():
    return "Tamil Classifier API is live."

if __name__ == "__main__":
    app.run()

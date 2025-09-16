from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, os, numpy as np

app = Flask(__name__)
CORS(app)

MODEL_PATH = "models/model.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Run train_model.py first to create model.joblib")

data = joblib.load(MODEL_PATH)
model, FEATURES = data["model"], data["features"]

def validate(payload):
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return None, f"Missing fields: {', '.join(missing)}"
    row = [payload[f] for f in FEATURES]
    return np.array(row).reshape(1, -1), None

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "JSON body required"}), 400
    X, err = validate(payload)
    if err:
        return jsonify({"error": err}), 400

    proba = model.predict_proba(X)[0,1]
    label = int(proba >= 0.5)
    band = "Low" if proba < 0.2 else "Moderate" if proba < 0.5 else "High"

    return jsonify({
        "probability": round(float(proba), 4),
        "risk_band": band,
        "label": label,
        "note": "Demo only. Not a medical diagnosis."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

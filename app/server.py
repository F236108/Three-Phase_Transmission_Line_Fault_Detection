"""
server.py  -  Flask backend for the Fault Detection Web Application.

Serves the professional HTML/CSS front end directly from app/index.html
and exposes a /predict API endpoint that returns predictions from both
the Logistic Regression (Set Q) model and the RBF SVM model.

Run:  python app/server.py
Then visit:  http://127.0.0.1:5000
"""

import sys, os
import json
import numpy as np
import joblib
from flask import Flask, send_file, request, jsonify, send_from_directory

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.logistic import sigmoid

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load models once at startup
# ---------------------------------------------------------------------------
RESULTS = os.path.join(ROOT, "results")
APP_DIR = os.path.dirname(os.path.abspath(__file__))

logistic_bundle = joblib.load(os.path.join(RESULTS, "logistic_Q_model.joblib"))
rbf_svm = joblib.load(os.path.join(RESULTS, "best_rbf_svm.joblib"))


def predict_logistic_q(raw_values):
    """Predict using Logistic Regression (Set Q) model."""
    theta = logistic_bundle["theta"]
    means = logistic_bundle["means"]
    stds  = logistic_bundle["stds"]

    raw = np.array(raw_values, dtype=np.float64)
    squared = raw ** 2
    features = np.concatenate([raw, squared])
    features_std = (features - means) / stds
    x = np.concatenate([[1.0], features_std])
    prob = float(sigmoid(x @ theta))
    label = 1 if prob >= 0.5 else 0
    return label, prob


def predict_rbf(raw_values):
    """Predict using RBF SVM model."""
    means_R = logistic_bundle["means"][:6]
    stds_R  = logistic_bundle["stds"][:6]

    raw = np.array(raw_values, dtype=np.float64)
    features_std = (raw - means_R) / stds_R
    label = int(rbf_svm.predict(features_std.reshape(1, -1))[0])
    return label


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Serve index.html directly from app/ directory."""
    index_path = os.path.join(APP_DIR, "index.html")
    return send_file(index_path)


@app.route("/results/<path:filename>")
def serve_results(filename):
    """Serve figures and tables from the results/ directory."""
    return send_from_directory(RESULTS, filename)


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint: receive 6 raw values, return predictions."""
    data = request.get_json(force=True)
    try:
        raw = [
            float(data["Ia"]), float(data["Ib"]), float(data["Ic"]),
            float(data["Va"]), float(data["Vb"]), float(data["Vc"]),
        ]
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    lr_label, lr_prob = predict_logistic_q(raw)
    svm_label = predict_rbf(raw)

    return jsonify({
        "logistic": {
            "label": lr_label,
            "probability": round(lr_prob, 6),
            "status": "FAULT DETECTED" if lr_label == 1 else "Normal Operation",
        },
        "svm": {
            "label": svm_label,
            "status": "FAULT DETECTED" if svm_label == 1 else "Normal Operation",
            "C": float(rbf_svm.C),
            "gamma": float(rbf_svm.gamma),
        },
    })


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Three-Phase Transmission Line Fault Detector Web App")
    print("  Server is live at: http://127.0.0.1:5000")
    print("  You can also double-click 'app/index.html' to open directly!")
    print("=" * 60 + "\n")
    app.run(debug=False, host="127.0.0.1", port=5000)

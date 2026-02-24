"""
app.py — MOTOAI Flask Backend
Exposes /predict endpoint for ML-powered car recommendation
and serves the frontend via Flask templates.
"""

import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the frontend HTML

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# ── Load model bundle at startup ─────────────────────────────────────────────
print("[MOTOAI] Loading ML model...")
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

clf = bundle["model"]
encoders = bundle["encoders"]
feature_cols = bundle["feature_cols"]
print(f"[MOTOAI] Model ready. Features: {feature_cols}")


# ── Helper: safely encode a value, fallback to 0 if unseen ──────────────────
def safe_encode(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        # Unknown label — use the closest known class (index 0)
        print(f"[WARN] Unknown label '{value}' — using fallback 0")
        return 0


# ── Mapping: UI filter strings → CSV column values ──────────────────────────
BODY_MAP = {
    "suv": "suv",
    "sedan": "sedan",
    "hatchback": "hatchback",
    "hatch": "hatchback",
    "electric": "electric",
    "ev": "electric",
    "luxury": "luxury",
    "muv": "suv",
    "mpv": "suv",
}

FUEL_MAP = {
    "petrol": "petrol",
    "diesel": "diesel",
    "electric": "electric",
    "cng": "cng",
    "hybrid": "petrol",  # map hybrid → petrol (closest)
}

BUDGET_MAP = {
    # Keys from the frontend filterState.budget strings
    "Under ₹6 Lakh": "under_6l",
    "₹6–10 Lakh": "6l_10l",
    "₹10–15 Lakh": "10l_15l",
    "₹15–25 Lakh": "15l_25l",
    "₹25–50 Lakh": "25l_50l",
    "₹50 Lakh+": "25l_50l",  # map 50L+ → highest known bucket
    # Also accept raw keys in case frontend sends them directly
    "under_6l": "under_6l",
    "6l_10l": "6l_10l",
    "10l_15l": "10l_15l",
    "15l_25l": "15l_25l",
    "25l_50l": "25l_50l",
}

SEATING_MAP = {
    "5 Seats": 5,
    "7 Seats": 7,
    "7+ Seats": 7,
    "5": 5,
    "7": 7,
}


# ── GET /  ── serve frontend ──────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("motoai.html")


# ── GET /health  ── health check ──────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "message": "MOTOAI ML Backend is running 🚗",
            "endpoints": ["/", "/predict", "/health"],
        }
    )


# ── POST /predict ─────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON body:
    {
        "body":    "SUV",           // body type filter
        "fuel":    "Petrol",        // fuel type filter
        "budget":  "₹10–15 Lakh",  // budget filter
        "seating": "5 Seats",       // seating filter
        "priority": "safety"        // optional: safety/comfort/performance/value
    }
    """
    try:
        data = request.get_json(force=True)

        # ── Parse & normalise inputs ─────────────────────────────
        raw_body = (data.get("body") or "suv").strip().lower()
        raw_fuel = (data.get("fuel") or "petrol").strip().lower()
        raw_budget = (data.get("budget") or "₹10–15 Lakh").strip()
        raw_seating = (data.get("seating") or "5 Seats").strip()
        priority = (data.get("priority") or "value").strip().lower()

        body_val = BODY_MAP.get(raw_body, "suv")
        fuel_val = FUEL_MAP.get(raw_fuel, "petrol")
        budget_val = BUDGET_MAP.get(raw_budget, "10l_15l")
        seating_val = SEATING_MAP.get(raw_seating, 5)

        print(
            f"[PREDICT] body={body_val} fuel={fuel_val} budget={budget_val}"
            f" seating={seating_val} priority={priority}"
        )

        # ── Priority → feature boosts ─────────────────────────────
        # Map user priority to high values for the chosen dimension
        priority_ratings = {
            "safety":      {"safety": 9, "comfort": 7, "performance": 7, "value": 7},
            "comfort":     {"safety": 7, "comfort": 9, "performance": 7, "value": 7},
            "performance": {"safety": 7, "comfort": 7, "performance": 9, "value": 7},
            "value":       {"safety": 7, "comfort": 7, "performance": 7, "value": 9},
            "default":     {"safety": 8, "comfort": 8, "performance": 8, "value": 8},
        }
        ratings = priority_ratings.get(priority, priority_ratings["default"])

        # ── Encode features ───────────────────────────────────────
        body_enc = safe_encode(encoders["body_type"], body_val)
        fuel_enc = safe_encode(encoders["fuel_type"], fuel_val)
        budget_enc = safe_encode(encoders["budget"], budget_val)

        feature_vector = np.array(
            [
                [
                    body_enc,
                    fuel_enc,
                    budget_enc,
                    seating_val,
                    ratings["safety"],
                    ratings["comfort"],
                    ratings["performance"],
                    ratings["value"],
                ]
            ]
        )

        # ── Predict: get top-3 probable classes ───────────────────
        proba = clf.predict_proba(feature_vector)[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        top3_cars = []
        target_le = encoders["recommended_car"]
        for idx in top3_idx:
            car_name = target_le.inverse_transform([idx])[0]
            confidence = round(float(proba[idx]) * 100, 1)
            top3_cars.append({"car": car_name, "confidence": confidence})

        # ── Response ──────────────────────────────────────────────
        response = {
            "status": "success",
            "recommended_car": top3_cars[0]["car"],
            "confidence": top3_cars[0]["confidence"],
            "top3": top3_cars,
            "inputs": {
                "body": body_val,
                "fuel": fuel_val,
                "budget": budget_val,
                "seating": seating_val,
                "priority": priority,
            },
        }
        print(f"[RESULT] {response['recommended_car']} ({response['confidence']}%)")
        return jsonify(response)

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

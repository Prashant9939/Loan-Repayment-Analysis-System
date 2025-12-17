# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)  # during development; restrict in production

# Load model and column order
model = joblib.load("loan_model.pkl")
with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

def encode_value(col, val):
    """
    Encode a single column value exactly as used in training.
    - For categorical columns convert to same numeric mapping.
    - For numeric columns convert to float, fallback 0.
    """
    if val is None:
        return 0

    s = str(val).strip()

    if col == "Married":
        return 1 if s.lower() == "yes" else 0

    if col == "Gender":
        return 1 if s.lower() == "male" else 0

    if col == "Self_Employed":
        return 1 if s.lower() == "yes" else 0

    if col == "Property_Area":
        if s.lower().startswith("u"):       # Urban
            return 2
        elif s.lower().startswith("s"):     # Semiurban
            return 1
        else:                               # Rural or default
            return 0

    if col == "Education":
        return 1 if s.lower().startswith("g") else 0  # Graduate vs Not Graduate

    if col == "Dependents":
        if s == "3+" or s.lower() == "3+":
            return 4    # same as training mapping
        # try numeric
        try:
            return int(s)
        except:
            return 0

    # numeric fallback
    try:
        return float(s)
    except:
        return 0.0

@app.route("/")
def home():
    return "Loan ML Model API Running Successfully"

@app.route("/predict", methods=["POST"])
def predict():
    # accept JSON body
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    # Build input vector in saved column order
    try:
        input_values = []
        for col in columns:
            raw = data.get(col, 0)
            encoded = encode_value(col, raw)
            input_values.append(encoded)

        arr = np.array(input_values).reshape(1, -1)
        pred = model.predict(arr)[0]

        # return integer 0/1 and human readable text
        return jsonify({
            "prediction": int(pred),
            "result": "Eligible" if int(pred) == 1 else "Not Eligible"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # debug=True for development; remove in production
    app.run(host="127.0.0.1", port=5000, debug=True)

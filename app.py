# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "models/model.pkl"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Modelo no encontrado. Ejecuta primero python -m src.main"}), 404

    model = joblib.load(MODEL_PATH)
    data = request.get_json()

    try:
        cliente = pd.DataFrame([data])[model.feature_names_in_]
        resultado = model.predict(cliente)[0]
        proba = model.predict_proba(cliente)[0]
        return jsonify({
            "churn": int(resultado),
            "prediccion": "Churn" if resultado == 1 else "No Churn",
            "probabilidad_churn": round(float(proba[1]) * 100, 1),
            "probabilidad_no_churn": round(float(proba[0]) * 100, 1)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)


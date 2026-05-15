"""
predict.py — Predicción sobre nuevos clientes
Rol 4: QA & Production Engineer

Uso:
    python -m src.predict
"""

import os
import sys
import joblib
import pandas as pd
import yaml


def load_model(model_path: str):
    """Carga el modelo entrenado desde disco."""
    if not os.path.exists(model_path):
        print(f"[ERROR] No se encontró el modelo en: {model_path}")
        print("  Asegúrate de haber ejecutado primero: python -m src.main")
        sys.exit(1)

    model = joblib.load(model_path)
    print(f"[INFO] Modelo cargado desde: {model_path}")
    return model


def predict_new_customer(model, customer: dict):
    """
    Predice si un cliente hará churn o no.

    Entrada: modelo entrenado + dict con features del cliente
    Salida: prediction (int), probability (float)
    """
    df = pd.DataFrame([customer])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return prediction, probability


def main():
    # 1. Cargar config
    config_path = "config/params.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_path = config["paths"]["model_save"]

    # 2. Cargar modelo
    model = load_model(model_path)

    # 3. Construir cliente de ejemplo usando las columnas exactas del modelo
    feature_names = model.feature_names_in_
    ejemplo_cliente = {col: 0 for col in feature_names}  # base en 0
    ejemplo_cliente.update({
        "gender": 1,
        "SeniorCitizen": 0,
        "Partner": 1,
        "tenure": 12,
        "MonthlyCharges": 75.5,
        "TotalCharges": 906.0,
        "PhoneService_Yes": 1,
        "InternetService_Fiber optic": 1,
        "PaperlessBilling_Yes": 1,
        "PaymentMethod_Electronic check": 1,
    })

    # 4. Predecir
    try:
        prediction, probability = predict_new_customer(model, ejemplo_cliente)
        resultado = "Churn" if prediction == 1 else "No Churn"
        print("\n[PREDICCIÓN]")
        print(f"  Resultado    : {resultado}")
        print(f"  Probabilidad : {probability:.2%}")
    except Exception as e:
        print(f"[ERROR] Fallo al predecir: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
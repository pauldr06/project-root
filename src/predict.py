# src/predict.py
import joblib
import pandas as pd
import os


def predict(model_path="models/model.pkl"):
    if not os.path.exists(model_path):
        print(f"[ERROR] No se encontró el modelo en: {model_path}")
        return

    model = joblib.load(model_path)

    # Cliente de ejemplo
    cliente = pd.DataFrame([{
        "gender": 1, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
        "tenure": 12, "PhoneService": 1, "MultipleLines": 0, "InternetService": 1,
        "OnlineSecurity": 0, "OnlineBackup": 1, "DeviceProtection": 0,
        "TechSupport": 0, "StreamingTV": 1, "StreamingMovies": 0,
        "Contract": 1, "PaperlessBilling": 1, "PaymentMethod": 2,
        "MonthlyCharges": 65.5, "TotalCharges": 786.0
    }])

    resultado = model.predict(cliente)[0]
    print(f"Predicción: {'Churn' if resultado == 1 else 'No Churn'}")


if __name__ == "__main__":
    predict()
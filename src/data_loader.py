"""
data_loader.py — Carga y preprocesamiento de datos
Rol 1: Data Engineer
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(config: dict):
    """
    Carga el CSV, limpia y preprocesa los datos.

    Entrada: config (dict cargado desde params.yaml)
    Salida: X_train, X_test, y_train, y_test
    """
    raw_path = config["data"]["raw_path"]
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    # 1. Cargar CSV
    df = pd.read_csv(raw_path)

    # 2. Limpiar TotalCharges: convertir a numérico, imputar con mediana
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # 3. Eliminar customerID
    df.drop(columns=["customerID"], inplace=True)

    # 4. Codificar columnas binarias a 0/1
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    df["Partner"] = df["Partner"].map({"Yes": 1, "No": 0})
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 5. One-hot encoding para el resto de columnas categóricas
    df = pd.get_dummies(df, drop_first=True)

    # 6. Separar features y target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 7. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
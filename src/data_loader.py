# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(config):
    df = pd.read_csv(config["paths"]["data_raw"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df = df.drop(columns=["customerID"])

    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.Categorical(df[col]).codes

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )
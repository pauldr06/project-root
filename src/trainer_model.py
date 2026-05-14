# src/trainer_model.py
# joblib es una librería de Python que sirve para guardar y cargar objetos grandes de forma eficiente
# de modelos de Machine Learning

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score


def train_and_save_model(X_train, y_train, X_test, y_test, config):
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=config["model"].get("n_estimators", 100),
            max_depth=config["model"].get("max_depth", None),
            random_state=config["data"]["random_state"]
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=config["model"].get("max_iter", 1000),
            random_state=config["data"]["random_state"]
        ),
        "SVM": SVC(
            C=config["model"].get("C", 1.0),
            random_state=config["data"]["random_state"]
        )
    }

    model_name = config["model"]["name"]
    if model_name not in models:
        raise ValueError(f"Modelo '{model_name}' no soportado. Opciones: {list(models)}")

    model = models[model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4)
    }

    joblib.dump(model, config["paths"]["model_save"])
    print(f"Modelo '{model_name}' guardado. Métricas: {metrics}")
    return metrics
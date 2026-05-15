"""
main.py — Orquestador del pipeline de MLOps
Rol 3: MLOps Engineer

Uso:
    python -m src.main
"""

import yaml
import os
import sys


def load_config(config_path: str = "config/params.yaml") -> dict:
    """Carga el archivo de configuración YAML."""
    if not os.path.exists(config_path):
        print(f"[ERROR] No se encontró el archivo de configuración: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"[INFO] Configuración cargada desde: {config_path}")
    return config


def main():
    print("=" * 50)
    print("  Pipeline MLOps - Predicción de Churn")
    print("=" * 50)

    # 1. Cargar configuración
    config = load_config()

    # 2. Etapa de datos (Rol 1: Data Engineer)
    print("\n[PASO 1] Cargando y preprocesando datos...")
    from src.data_loader import load_and_preprocess_data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    # 3. Etapa de entrenamiento (Rol 2: ML Engineer)
    print(f"\n[PASO 2] Entrenando modelo: {config['model']['name']}...")
    from src.trainer_model import train_and_save_model
    metrics = train_and_save_model(X_train, y_train, X_test, y_test, config)

    # 4. Mostrar resultados
    print("\n[RESULTADOS]")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1_score']:.4f}")
    print(f"\n  Modelo guardado en: {config['paths']['model_save']}")
    print("=" * 50)
    print("  Pipeline completado exitosamente.")
    print("=" * 50)


if __name__ == "__main__":
    main()
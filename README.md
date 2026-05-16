# 📡 Proyecto Colaborativo MLOps: Predicción de Churn (Abandono de Clientes)

## 🎯 Objetivo del Proyecto
Construir un pipeline de Machine Learning modular, reproducible y colaborativo para predecir si un cliente de telecomunicaciones abandonará el servicio (**Churn**).

El proyecto simula un entorno laboral real donde **4 roles especializados** deben integrar su código en un solo repositorio usando Git.

---

## 📂 El Dataset
Todos los equipos trabajarán con el dataset **Telco Customer Churn**.

*   **Fuente:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
*   **Archivo:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
*   **Problema:** Clasificación Binaria (¿El cliente se va? `Yes`/`No`)
*   **Instrucción Importante:**
    1.  Descarguen el CSV.
    2.  Guárdenlo en la carpeta `data/raw/`.
    3.  **NO suban el CSV a Git** (ya está configurado en `.gitignore`).

---

## 👥 Roles y Responsabilidades (Equipos de 4)

### 1. 👷 Data Engineer (`src/data_loader.py`)
**Tu misión:** Transformar datos brutos y sucios en datos limpios listos para entrenar.

*   Cargar el CSV desde `data/raw/`.
*   Limpiar `TotalCharges`, eliminar `customerID`, codificar variables categóricas.
*   Dividir en Train/Test usando `config/params.yaml`.
*   **Entregable:** Función `load_and_preprocess_data(config)` → `X_train, X_test, y_train, y_test`.

### 2. 🧠 ML Engineer (`src/trainer_model.py`)
**Tu misión:** Experimentar con algoritmos y guardar el mejor modelo.

*   Fábrica de modelos: `RandomForest`, `LogisticRegression`, `SVM`.
*   Calcular métricas: Accuracy, Recall y F1-Score.
*   Guardar el modelo en `models/` usando `joblib`.
*   **Entregable:** Función `train_and_save_model(X_train, y_train, X_test, y_test, config)` → diccionario de métricas.

### 3. ⚙️ MLOps Engineer (`src/main.py` y `config/`)
**Tu misión:** Orquestar el flujo y gestionar la configuración externa.

*   Crear y mantener `config/params.yaml`.
*   Escribir `src/main.py` que ejecute el pipeline completo.
*   **Entregable:** `python -m src.main` corre sin errores.

### 4. 🛡️ QA & Production Engineer (`src/predict.py` y `tests/`)
**Tu misión:** Validar que el sistema funcione y preparar la inferencia.

*   `src/predict.py`: carga el modelo y predice sobre un cliente de ejemplo.
*   `tests/test_pipeline.py`: al menos 2 tests unitarios.
*   **Entregable:** Script de predicción robusto y tests pasando.

---

## 🚀 Flujo de Trabajo con Git

1.  **Clonar:** `git clone <url-del-repo-del-equipo>`
2.  **Ramas:** Cada alumno crea su rama:
    *   `git checkout -b feature/data-engineer`
    *   `git checkout -b feature/ml-engineer`
    *   `git checkout -b feature/mlops-engineer`
    *   `git checkout -b feature/qa-engineer`
3.  **Desarrollo:** Trabajen en paralelo con commits frecuentes.
4.  **Integración:** El MLOps Engineer mergea todas las ramas a `main`.
5.  **Prueba Final:** `python -m src.main` en rama `main`.

---

## 📂 Estructura de Carpetas

```text
churn-mlops-project/
├── config/
│   └── params.yaml
├── data/
│   └── raw/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── trainer_model.py
│   ├── main.py
│   └── predict.py
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
├── models/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Resultados del Mejor Modelo

| Métrica   | Valor  |
|-----------|--------|
| Accuracy  | 0.8062 |
| Recall    | 0.5147 |
| F1 Score  | 0.5845 |

**Modelo:** RandomForest (`n_estimators=100`, `max_depth=10`)

---

## 🤖 LLM Contribution

| Rol | Herramienta | Uso |
|-----|-------------|-----|
| ML Engineer (Emilio) | Claude | Estructurar la fábrica de modelos, depurar errores de preprocesamiento y resolver conflictos de Git |
| Data Engineer (Mateo) | - | - |
| MLOps Engineer | - | - |
| QA Engineer | - | - |

---

## ✅ Checklist de Entrega

*   [x] El comando `python -m src.main` ejecuta todo el pipeline sin errores.
*   [x] El archivo `config/params.yaml` existe y controla los hiperparámetros.
*   [x] Hay al menos 2 modelos diferentes implementados en el código.
*   [x] El script `predict.py` carga el modelo y hace una predicción de ejemplo.
*   [x] El historial de Git muestra contribuciones de los 4 miembros del equipo.
*   [x] El `README.md` final incluye los resultados obtenidos (Accuracy/Recall del mejor modelo).

---

¡Éxito con la clase! Es un ejercicio excelente para ver quién realmente entiende la integración de sistemas. 🚀
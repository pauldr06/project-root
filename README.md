# Proyecto MLOps: Predicción de Churn en Telecomunicaciones

## ¿Qué hace este proyecto?
Pipeline de Machine Learning para predecir si un cliente de telecomunicaciones abandonará el servicio (Churn). Construido de forma colaborativa con 4 roles especializados simulando un entorno laboral real.

---

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/pauldr06/project-root.git
cd project-root
```

2. Crea un entorno virtual e instala dependencias:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. Descarga el dataset desde [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn) y coloca el archivo `WA_Fn-UseC_-Telco-Customer-Churn.csv` en `data/raw/`.

---

## Uso

Ejecutar el pipeline completo:
```bash
python -m src.main
```

Hacer una predicción sobre un cliente de ejemplo:
```bash
python src/predict.py
```

Correr los tests:
```bash
pytest test/test_pipeline.py -v
```

Levantar la API:
```bash
python app.py
```

Ejemplo de salida del pipeline:

Pipeline MLOps - Predicción de Churn
[PASO 1] Cargando y preprocesando datos...
Train: (5634, 19) | Test: (1409, 19)
[PASO 2] Entrenando modelo: RandomForest...
Accuracy  : 0.8062
Recall    : 0.5147
F1 Score  : 0.5845

---

## Dataset

- **Fuente:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Archivo:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Problema:** Clasificación binaria, predecir si un cliente hace Churn (Yes/No)
- **Importante:** No subir el CSV a Git, está en `.gitignore`

---

## Roles y Responsabilidades

### Data Engineer (`src/data_loader.py`)
Carga el CSV, limpia `TotalCharges`, elimina `customerID`, codifica variables categóricas y divide en Train/Test.
Entregable: `load_and_preprocess_data(config)` → `X_train, X_test, y_train, y_test`

### ML Engineer (`src/trainer_model.py`)
Implementa una fábrica de modelos (RandomForest, LogisticRegression, SVM), entrena, calcula métricas y guarda el modelo con joblib.
Entregable: `train_and_save_model(X_train, y_train, X_test, y_test, config)` → diccionario de métricas

### MLOps Engineer (`src/main.py` y `config/`)
Mantiene `params.yaml` y orquesta el pipeline completo en `main.py`.
Entregable: `python -m src.main` corre sin errores

### QA & Production Engineer (`src/predict.py` y `tests/`)
Crea el script de predicción con manejo de errores y los tests unitarios.
Entregable: script de predicción funcional y 2 tests pasando

---

## Flujo de trabajo con Git

1. Clonar el repo
2. Cada miembro crea su rama (`feature/data-engineer`, `feature/ml-engineer`, etc.)
3. Desarrollar y hacer commits frecuentes
4. El MLOps Engineer mergea todo a `main`
5. Prueba final: `python -m src.main`

---

## API REST

El proyecto incluye una API Flask para hacer predicciones desde cualquier cliente HTTP o desde el navegador.

Levantar el servidor:
```bash
python app.py
# Disponible en http://localhost:5000
```

**GET** `/` — Abre una interfaz web con formulario para probar el modelo sin escribir código. Muestra la probabilidad de churn, barra visual y un resumen con los datos clave del cliente.

**POST** `/predict` — Endpoint JSON para integraciones.

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":1,"SeniorCitizen":0,"Partner":1,"Dependents":0,"tenure":12,"PhoneService":1,"MultipleLines":0,"InternetService":1,"OnlineSecurity":0,"OnlineBackup":1,"DeviceProtection":0,"TechSupport":0,"StreamingTV":1,"StreamingMovies":0,"Contract":1,"PaperlessBilling":1,"PaymentMethod":2,"MonthlyCharges":65.5,"TotalCharges":786.0}'
```

Respuesta:
```json
{
  "churn": 0,
  "prediccion": "No Churn",
  "probabilidad_churn": 18.5,
  "probabilidad_no_churn": 81.5
}
```

> Las variables categóricas van codificadas como enteros porque el modelo fue entrenado con `pd.Categorical().codes`. Ver `src/data_loader.py` para el mapeo exacto.

---

## Estructura del proyecto

```text
project-root/
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
├── templates/
│   └── index.html
├── test/
│   ├── __init__.py
│   └── test_pipeline.py
├── models/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Resultados del mejor modelo

| Métrica   | Valor  |
|-----------|--------|
| Accuracy  | 0.8062 |
| Recall    | 0.5147 |
| F1 Score  | 0.5845 |

Modelo: RandomForest (`n_estimators=100`, `max_depth=10`)

---

## LLM Contribution

| Rol | Herramienta | Uso |
|-----|-------------|-----|
| ML Engineer (Emilio) | Claude | Estructurar la fábrica de modelos, depurar errores de preprocesamiento y resolver conflictos de Git |
| Data Engineer (Mateo) | - | - |
| MLOps Engineer | - | - |
| QA Engineer | - | - |

---

## Checklist de entrega

- [x] `python -m src.main` ejecuta el pipeline sin errores
- [x] `config/params.yaml` controla los hiperparámetros
- [x] Al menos 2 modelos implementados
- [x] `predict.py` hace una predicción de ejemplo
- [x] Historial de Git con contribuciones de los 4 miembros
- [x] README con Accuracy y Recall del mejor modelo
- [x] API REST con interfaz web para probar el modelo


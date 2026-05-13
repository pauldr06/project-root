from data_loader import load_and_clean_data

if __name__ == "__main__":
    ruta = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    print("Iniciando el pipeline...")
    
    # Llamamos a la función que escribiste en el otro archivo
    df = load_and_clean_data(ruta)
    
    # Guardamos
    df.to_csv("data/processed/data_cleaned.csv", index=False)
    print("¡Proceso terminado desde main.py!")
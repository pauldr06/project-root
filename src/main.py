from data_loader import load_and_clean_data
#Este bloque te entrega el dataset listo y se guarda en data/processed/data_cleaned.csv. :)) P
if __name__ == "__main__":
    ruta = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_and_clean_data(ruta)
    df.to_csv("data/processed/data_cleaned.csv", index=False)
    print("¡Proceso terminado!")
from data_loader import load_and_clean_data
from sklearn.model_selection import train_test_split
#Este bloque te entrega el dataset listo y se guarda en data/processed/data_cleaned.csv. :)) P
if __name__ == "__main__":
    ruta = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_and_clean_data(ruta)
    
    train, test = train_test_split(df,test_size=0.2, random_state=42)
    test.to_csv("data/processed/data_cleaned.csv", index=False)
    train.to_csv("data/processed/train.csv", index=False)

    print(f"¡Éxito! Creados train.csv ({len(train)} filas) y test.csv ({len(test)} filas)")
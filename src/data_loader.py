import pandas as pd
import numpy as np
import yaml
def load_and_clean_data(input_path):
		df = pd.read_csv(input_path)
		df['TotalCharges']= pd.to_numeric(df['TotalCharges'], errors='coerce')
		median_value = df['TotalCharges'].median()
		df['TotalCharges'] = df['TotalCharges'].fillna(median_value)
		return df 


#def load_data(): 
#	pass



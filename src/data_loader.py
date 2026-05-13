import pandas as pd
import numpy as np


def load_and_clean_data(input_path):
		df = pd.read_csv(input_path)
		df['TotalCharges']= pd.to_numeric(df['TotalCharges'], errors='coerce')
		median_value = df['TotalCharges'].median()
		df['TotalCharges'] = df['TotalCharges'].fillna(median_value)
		

		if 'customerID' in df.columns:
			df = df.drop(columns=['customerID'])

		# Gender: Female=0, Male=1 
		# Partner/Churn: No=0, Yes=1

		binary_map={'Yes':1, 'No':0}
		gender_map={'Female':0, 'Male':1}

		df['gender'] = df['gender'].map(gender_map)
		df['Partner'] = df['Partner'].map(binary_map)
		df['Churn'] = df['Churn'].map(binary_map)

		return df 



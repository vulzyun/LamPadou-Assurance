import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('car_insurance.csv')
'''
print(df.shape) #dimensions du jeu de données (colonnes et lignes)
print("==========================================================")
print(df.dtypes) #Affichage des types de données
print("==========================================================")
print(df.isna) #Affichage des données 'Not Available'
print("==========================================================")
print(df.hist()) #Affichage des statistiques descriptives
print("==========================================================")
print(df.isna().sum()) #Affichage des données 'Not Available' par colonne
print("==========================================================")
'''



def changedf(df):
    df_copy = df.copy()
    children_median = df_copy[' children'].median()
    mileage_median = df_copy[' annual_mileage'].median()
    speeding_median = df_copy['speeding_violations'].median()
    accidents_median = df_copy['past_accidents'].median()
    
    for column in df_copy.columns:
        if df_copy[column].dtype in ['int64', 'float64']:
            median_value = df_copy[column].median()
            df_copy[column] = df_copy[column].fillna(median_value)
        else:
            mode_value = df_copy[column].mode()[0] if not df_copy[column].mode().empty else None
            df_copy[column] = df_copy[column].fillna(mode_value)
        
        if column == 'children':
            df_copy[column] = df_copy[column].fillna(0)
            df_copy.loc[df_copy[column] > 12, column] = children_median
        
        elif column == 'annual_mileage':
            df_copy.loc[df_copy[column] > 20000, column] = mileage_median
        
        elif column == 'speeding_violations':
            df_copy.loc[df_copy[column] > 12, column] = speeding_median
        
        elif column == 'past_accidents':
            df_copy.loc[df_copy[column] > 12, column] = accidents_median
    
    return df_copy



#print(df.loc[:,'children'].tolist())
print(df.columns)
print(df[' gender'])
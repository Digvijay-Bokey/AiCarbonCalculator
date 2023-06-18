import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
from sklearn.impute import KNNImputer

# Predefine selected columns from original dataset
selected_columns = ['REGIONC', 'DIVISION', 'state_name', 'BA_climate', 'TYPEHUQ', 'YEARMADERANGE', 'WALLTYPE', 'SWIMPOOL', 'RECBATH', 'FUELTUB', 'RANGEFUEL', 'OUTGRILLFUEL', 'DWASHUSE', 'DRYRFUEL', 'EQUIPM', 'FUELHEAT', 'FUELH2O', 'MONEYPY']

# Read data
original_df = pd.read_csv('recs2020_public_v3.csv')

# Create new dataframe with selected columns and the condition
new_df = original_df.head(2000)[selected_columns]

# Filter rows based on the condition (hot tub or pool)
condition = (new_df['SWIMPOOL'] == 1) | (new_df['RECBATH'] == 1)
new_df = new_df[condition]

# Impute missing values using KNN imputation
imputer = KNNImputer(n_neighbors=5)
new_df_imputed = pd.DataFrame(imputer.fit_transform(new_df), columns=new_df.columns)

def calculate_carbon_footprint(row):
    # Get the values from the row
    values = row[['TYPEHUQ', 'YEARMADERANGE', 'WALLTYPE', 'SWIMPOOL', 'RECBATH', 'FUELTUB', 'RANGEFUEL', 'OUTGRILLFUEL', 'DWASHUSE', 'DRYRFUEL', 'EQUIPM', 'FUELHEAT', 'FUELH2O', 'MONEYPY']]
    
    # Preprocess the values (handle zeros, log transformation, etc.)
    transformed_values = []
    for value in values:
        if value <= 0:
            transformed_values.append(np.nan)
        else:
            transformed_values.append(np.log1p(value))
    
    # Create the SARIMAX model with the desired parameters
    model = sm.tsa.SARIMAX(transformed_values, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12))
    
    # Fit the model to the values
    result = model.fit()
    
    # Predict the next value
    prediction = result.predict(start=len(transformed_values), end=len(transformed_values))
    
    # Calculate the total carbon footprint
    total_carbon_footprint = np.expm1(prediction)
    
    return total_carbon_footprint

# Apply the calculate_carbon_footprint function to each row in new_df_imputed
new_df_imputed['CARBFTP'] = new_df_imputed.apply(calculate_carbon_footprint, axis=1)

# Save new dataframe to csv
new_df_imputed.to_csv('selected_with_carbon_footprint.csv', index=False)

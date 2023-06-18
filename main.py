import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm

# Predefine selected columns from original dataset
selected_columns = ['REGIONC', 'DIVISION', 'state_name', 'BA_climate', 'TYPEHUQ', 'YEARMADERANGE', 'WALLTYPE', 'SWIMPOOL', 'RECBATH', 'FUELTUB', 'RANGEFUEL', 'OUTGRILLFUEL', 'DWASHUSE', 'DRYRFUEL', 'EQUIPM', 'FUELHEAT', 'FUELH2O', 'MONEYPY']

# Read data
original_df = pd.read_csv('recs2020_public_v3.csv')

# Create new dataframe with selected columns and the condition
new_df = original_df[selected_columns]

# Filter rows based on the condition (hot tub or pool)
condition = (new_df['SWIMPOOL'] == 1) | (new_df['RECBATH'] == 1)
new_df = new_df[condition].head(50)

# Change negative values and zeros to NaN for numeric columns
numeric_columns = new_df.select_dtypes(include=np.number).columns
new_df[numeric_columns] = new_df[numeric_columns].where(new_df[numeric_columns] > 0, np.nan)

def calculate_carbon_footprint(row):
    # Get the values from the row
    values = row[['TYPEHUQ', 'YEARMADERANGE', 'WALLTYPE', 'SWIMPOOL', 'RECBATH', 'FUELTUB', 'RANGEFUEL', 'OUTGRILLFUEL', 'DWASHUSE', 'DRYRFUEL', 'EQUIPM', 'FUELHEAT', 'FUELH2O', 'MONEYPY']]
    
    # Preprocess the values (handle NaNs, log transformation, etc.)
    transformed_values = []
    for value in values:
        if pd.isna(value):
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

# Apply the calculate_carbon_footprint function to each row in new_df
new_df['CARBFTP'] = new_df.apply(calculate_carbon_footprint, axis=1)

# Save new dataframe to csv
new_df.to_csv('selected_with_carbon_footprint.csv', index=False)

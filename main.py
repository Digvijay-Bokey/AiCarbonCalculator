import pandas as pd
import numpy as np
import statsmodels.api as sm

# Predefine selected columns from original dataset
selected_columns = ['REGIONC', 'DIVISION', 'state_name', 'BA_climate', 'TYPEHUQ', 'YEARMADERANGE', 'WALLTYPE', 'SWIMPOOL', 'RECBATH', 'FUELTUB', 'RANGEFUEL', 'OUTGRILLFUEL', 'DWASHUSE', 'DRYRFUEL', 'EQUIPM', 'FUELHEAT', 'FUELH2O', 'MONEYPY']

# Read data
original_df = pd.read_csv('recs2020_public_v3.csv')

# Create new dataframe with selected columns and the condition
new_df = original_df[selected_columns]

# Filter rows based on the condition (hot tub or pool)
condition = (new_df['SWIMPOOL'] == 1) | (new_df['RECBATH'] == 1)
new_df = new_df[condition]

# Calculate the Mean Squared Error (MSE) for each row
def calculate_mse(row):
    # Get relevant data points and handle zeros
    data = row[['TYPEHUQ', 'YEARMADERANGE', 'WALLTYPE', 'SWIMPOOL', 'RECBATH', 'FUELTUB', 'RANGEFUEL', 'OUTGRILLFUEL', 'DWASHUSE', 'DRYRFUEL', 'EQUIPM', 'FUELHEAT', 'FUELH2O', 'MONEYPY']]
    numeric_columns = data.select_dtypes(include=np.number).columns
    data[numeric_columns] = data[numeric_columns].apply(lambda x: np.log1p(x) if x > 0 else x)
    
    # Create SARIMA model and fit to data
    model = sm.tsa.SARIMAX(data, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12))
    result = model.fit()
    
    # Predict the next value
    prediction = result.predict(start=data.index[-1], end=data.index[-1] + pd.DateOffset(months=1))
    
    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((np.expm1(prediction) - np.expm1(data.iloc[-1])) ** 2)
    
    return mse

# Add a new column to selected.csv with the Mean Squared Error (MSE) for each row
new_df['MSE'] = new_df.apply(calculate_mse, axis=1)

# Save new dataframe to csv
new_df.to_csv('selected_with_mse.csv', index=False)

print(new_df)
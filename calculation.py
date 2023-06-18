import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm

# Calculate carbon footprint for each data point using SARIMA model
def calculate_score(row):
    total_carbon_footprint = row['TYPEHUQ'] + row['YEARMADERANGE'] + row['WALLTYPE']

    return total_carbon_footprint

# Read the selected.csv file
selected_df = pd.read_csv('selected.csv')

# Apply function to each row and add the results as a new column
selected_df['CARBFTP'] = selected_df.apply(calculate_score, axis=1)

# Save the updated dataframe to the selected.csv file
selected_df.to_csv('selected.csv', index=False)

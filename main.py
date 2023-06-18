import pandas as pd

# Read data
original_df = pd.read_csv('recs2020_public_v3.csv')

# Create new dataframe with selected columns
selected_columns = ['REGIONC']
new_df = original_df[selected_columns]

# Save new dataframe to csv
new_df.to_csv('selected.csv', index=False)

print(new_df)
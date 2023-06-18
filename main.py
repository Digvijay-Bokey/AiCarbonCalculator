import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler

# Predefine selected columns from original dataset
selected_columns = ['REGIONC', 'DIVISION', 'state_name', 'BA_climate', 'TYPEHUQ', 'YEARMADERANGE', 'WALLTYPE', 'SWIMPOOL', 'RECBATH', 'FUELTUB', 'RANGEFUEL', 'OUTGRILLFUEL', 'DWASHUSE', 'DRYRFUEL', 'EQUIPM', 'FUELHEAT', 'FUELH2O', 'MONEYPY']

# Read data from csv file into dataframe
original_df = pd.read_csv('recs2020_public_v3.csv')

# Create new dataframe with selected columns and the condition
new_df = original_df[selected_columns]

le = preprocessing.LabelEncoder()

new_df['REGIONC'] = le.fit_transform(new_df['REGIONC'])
new_df['DIVISION'] = le.fit_transform(new_df['DIVISION'])
new_df['state_name'] = le.fit_transform(new_df['state_name'])
new_df['BA_climate'] = le.fit_transform(new_df['BA_climate'])

# Filter rows based on the condition (hot tub or pool)
condition = (new_df['SWIMPOOL'] == 1) | (new_df['RECBATH'] == 1)
new_df = new_df[condition]

# Change negative values and zeros to NaN for numeric columns
#numeric_columns = new_df.select_dtypes(include=np.number).columns
#new_df[numeric_columns] = new_df[numeric_columns].where(new_df[numeric_columns] > 0, np.nan)

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
    model = sm.tsa.SARIMAX(transformed_values, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12), enforce_stationarity=False)
    
    # Fit the model to the values
    result = model.fit()
    
    # Predict the next value
    prediction = result.predict(start=len(transformed_values), end=len(transformed_values))
    
    # Calculate the total carbon footprint
    total_carbon_footprint = np.expm1(prediction)
    
    return total_carbon_footprint

# Apply the calculate_carbon_footprint function to each row in new_df
new_df['CARBFTP'] = new_df.apply(calculate_carbon_footprint, axis=1)

# Cap carbon footprint to 15
new_df['CARBFTP'][new_df['CARBFTP'] > 15] = 15

# Scale the carbon footprint to 100
new_df.loc[:, 'CARBFTP'] = new_df['CARBFTP'] / 15 * 100

# Divide dataset into training and testing datasets
dataset = new_df.values
X = dataset[:, 0:18]
Y = dataset[:, 18]

X_scale = preprocessing.MinMaxScaler().fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.2)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

"""
import tensorflow as tf 
from tensorflow.python.keras.layers import Input, Dense 

from keras.models import Sequential
from keras.layers import Dense

model = Sequential([
    Dense(32, activation='relu', input_shape = (12,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])
                   
model.compile(optimizer='sgd',
              loss = 'binary_crossentropy',
              metrics=['accuracy'])


hist = model.fit(X_train, Y_test, 
                 batch_size=32, epochs=100,
                 validation_data=(X_val, Y_val))

"""

# Create and train the MLPRegressor model
mlp = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', random_state=42)
mlp.fit(X_train, Y_train)

# Make predictions on the test set
pred = mlp.predict(X_test)

# Evaluate the model using Mean Absolute Error (MAE)
mae = metrics.mean_absolute_error(Y_test, pred)
print("Mean Absolute Error:", mae)

# Evaluate the model using Mean Squared Error (MSE)
mse = metrics.mean_squared_error(Y_test, pred)
print("Mean Squared Error:", mse)

# Evaluate the model using R2 score
r2 = metrics.r2_score(Y_test, pred)
print("Accuracy Score:", r2)

#test = ['WEST', 'Pacific', 'California', 'Hot-Dry', 2, 1, 4, 1, 0, -2, 1, 1, 5, 1, 3, 1, 1, 16, 7]

test_df = pd.DataFrame(columns=['REGIONC','DIVISION','state_name','BA_climate', 'TYPEHUQ', 'YEARMADERANGE', 'WALLTYPE',
                                'SWIMPOOL', 'RECBATH', 'FUELTUB', 'RANGEFUEL', 'OUTGRILLFUEL', 'DWASHUSE', 'DRYRFUEL',
                                'EQUIPM', 'FUELHEAT', 'FUELH2O', 'MONEYPY'], index = ['x'])


test_df['REGIONC'] = pd.Series({'x':'West'})
test_df['DIVISION'] = pd.Series({'x':'Pacific'})
test_df['state_name'] = pd.Series({'x':'California'})
test_df['BA_climate'] = pd.Series({'x':'Hot-Dry'})
test_df['TYPEHUQ'] = pd.Series({'x':2})
test_df['YEARMADERANGE'] = pd.Series({'x':1})
test_df['WALLTYPE'] = pd.Series({'x':4})
test_df['SWIMPOOL'] = pd.Series({'x':1})
test_df['RECBATH'] = pd.Series({'x':0})
test_df['FUELTUB'] = pd.Series({'x':-2})
test_df['RANGEFUEL'] = pd.Series({'x':1})
test_df['OUTGRILLFUEL'] = pd.Series({'x':1})
test_df['DWASHUSE'] = pd.Series({'x':5})
test_df['DRYRFUEL'] = pd.Series({'x':8})
test_df['EQUIPM'] = pd.Series({'x':3})
test_df['FUELHEAT'] = pd.Series({'x':4})
test_df['FUELH2O'] = pd.Series({'x':1})
test_df['MONEYPY'] = pd.Series({'x':7})


le = preprocessing.LabelEncoder()

test_df['REGIONC'] = le.fit_transform(test_df['REGIONC'])
test_df['DIVISION'] = le.fit_transform(test_df['DIVISION'])
test_df['state_name'] = le.fit_transform(test_df['state_name'])
test_df['BA_climate'] = le.fit_transform(test_df['BA_climate'])


preds = preprocessing.MinMaxScaler().fit_transform(test_df)
final = mlp.predict(preds)

print("PREDICTED CARBON VAL:")
print(final)

# Save new dataframe to csv
new_df.to_csv('selected_with_carbon_footprint.csv', index=False)



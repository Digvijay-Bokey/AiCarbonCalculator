from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.metrics import confusion_matrix

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def carbon_predict():
    if request.method == 'POST':
        form_data = {
            'REGIONC': request.form.get('region'),
            'DIVISION': request.form.get('specific-region'),
            'state_name': request.form.get('state'),
            'BA_climate': request.form.get('climate'),
            'TYPEHUQ': int(request.form.get('housing')),
            'YEARMADERANGE': int(request.form.get('house-built')),
            'WALLTYPE': int(request.form.get('wall-type')),
            'SWIMPOOL': int(request.form.get('swimming-pool')),
            'RECBATH': int(request.form.get('hot-tub')),
            'FUELTUB': int(request.form.get('hot-tub-fuel')),
            'RANGEFUEL': int(request.form.get('range-fuel')),
            'OUTGRILLFUEL': int(request.form.get('grill-fuel')),
            'DWASHUSE': int(request.form.get('dishwasher-use')),
            'DRYRFUEL': int(request.form.get('clothes-dryer-fuel')),
            'EQUIPM': int(request.form.get('main-space-equipment')),
            'FUELHEAT': int(request.form.get('main-space-heating-fuel')),
            'FUELH2O': int(request.form.get('main-water-heater-fuel')),
            'MONEYPY': int(request.form.get('annual-income')),
        }
        # Pass this form_data to your ML model and get the result.
        result = main(form_data) 
        return render_template('results.html', 
                       mae=result['mae'], 
                       mse=result['mse'], 
                       accuracy=result['accuracy'],
                       final=result['final'], 
                       fpr=result['fpr'], 
                       tnr=result['tnr'], 
                       tpr=result['tpr'], 
                       fnr=result['fnr'])

    else:
        return render_template('form.html')

def main(form_data):

    # Features that we will be using to calculate "carbon score" for users
    selected_columns = ['REGIONC', 'DIVISION', 'state_name', 'BA_climate', 'TYPEHUQ', 'YEARMADERANGE',
                        'WALLTYPE', 'SWIMPOOL', 'RECBATH', 'FUELTUB', 'RANGEFUEL', 'OUTGRILLFUEL',
                        'DWASHUSE', 'DRYRFUEL', 'EQUIPM', 'FUELHEAT', 'FUELH2O', 'MONEYPY']

    # Read data from csv file into dataframe
    original_df = pd.read_csv('recs2020_public_v3.csv')

    # Create new dataframe with selected columns and the condition
    new_df = original_df[selected_columns]

    # convert Strings to ASCII, so comparisons can adequately be made
    le = preprocessing.LabelEncoder()

    new_df['REGIONC'] = le.fit_transform(new_df['REGIONC'])
    new_df['DIVISION'] = le.fit_transform(new_df['DIVISION'])
    new_df['state_name'] = le.fit_transform(new_df['state_name'])
    new_df['BA_climate'] = le.fit_transform(new_df['BA_climate'])

    # Filter rows based on the condition (hot tub or pool)
    # if user has either we state that the house has a water unit (represented by column condition"
    condition = (new_df['SWIMPOOL'] == 1) | (new_df['RECBATH'] == 1)
    new_df = new_df[condition]

    # Change negative values and zeros to NaN for numeric columns
    # we arrived at better testing accuracy without imputing data and rather just left missing data as -2 as in original dataset
    # numeric_columns = new_df.select_dtypes(include=np.number).columns
    # new_df[numeric_columns] = new_df[numeric_columns].where(new_df[numeric_columns] > 0, np.nan)

    # calculates carbon footprint using features using SARIMA (moving averages)
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
        
        # Predict the carbpn footprint with moving average methodology
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

    # Save new dataframe to csv
    new_df.to_csv('selected_with_carbon_footprint.csv', index=False)

    # Divide dataset into training and testing datasets
    dataset = new_df.values
    X = dataset[:, 0:18]
    Y = dataset[:, 18]

    X_scale = preprocessing.MinMaxScaler().fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.2)
    # print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    # Create and train the MLPRegressor model to generate AI predictions
    model = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', random_state=42)
    model.fit(X_train, Y_train)

    # Make predictions on the test set
    pred = model.predict(X_test)

    # Evaluate the model using Mean Absolute Error (MAE)
    mae = metrics.mean_absolute_error(Y_test, pred)
    # print("Mean Absolute Error:", mae)

    # Evaluate the model using Mean Squared Error (MSE)
    mse = metrics.mean_squared_error(Y_test, pred)
    # print("Mean Squared Error:", mse)

    # Evaluate the model using an accuracy score
    accuracy = metrics.r2_score(Y_test, pred)
    # print("Accuracy Score:", accuracy)


    # create a user input as a sole data row calling it test_df
    test_df = pd.DataFrame(columns=['REGIONC','DIVISION','state_name','BA_climate', 'TYPEHUQ', 'YEARMADERANGE', 'WALLTYPE',
                                    'SWIMPOOL', 'RECBATH', 'FUELTUB', 'RANGEFUEL', 'OUTGRILLFUEL', 'DWASHUSE', 'DRYRFUEL',
                                    'EQUIPM', 'FUELHEAT', 'FUELH2O', 'MONEYPY'], index = ['x'])


    # input user inputs from front end to the test dataframe
    test_df.loc['x'] = pd.Series(form_data)

    # change strings to ASCII conversion as done in preprocessing stage
    le = preprocessing.LabelEncoder()

    test_df['REGIONC'] = le.fit_transform(test_df['REGIONC'])
    test_df['DIVISION'] = le.fit_transform(test_df['DIVISION'])
    test_df['state_name'] = le.fit_transform(test_df['state_name'])
    test_df['BA_climate'] = le.fit_transform(test_df['BA_climate'])

    def runMLP_Regressor(data):
        # Preprocess input data here, e.g. turn data into appropriate numpy array
        # Use your model to make a prediction
        # head limit used to be here
        prediction = model.predict(data)


        # Post-process prediction here, if necessary
        return prediction

    #truncate carbon score prediction to two three decimal places
    def threeDecimalPlaces(x):
        return int(x*1000)/1000
    # find carbon score for this user using the AI model we trained (the MLPRegressor which we named model)
    preds = preprocessing.MinMaxScaler().fit_transform(test_df)
    final = runMLP_Regressor(preds)/15 * 100  # scale to 100
    final = threeDecimalPlaces(final)

    # outputs predicted score for user (send to backend
    # print("PREDICTED CARBON SCORE: " + str(final))

    #binVal being 1 means user is responsible
    #binVal being 0 means user is not responsible
    binVal = 1 if final > 50 else 0


    # evaluating fairness
    fairness_df = new_df
    fairness_df.loc[fairness_df["CARBFTP"] >= 50, "y_test"] = 1
    fairness_df.loc[fairness_df["CARBFTP"] < 50, "y_test"] = 0

    fairness_df.loc[fairness_df["YEARMADERANGE"] >= 5, "y_true"] = 1
    fairness_df.loc[fairness_df["YEARMADERANGE"] < 5, "y_true"] = 0

    # Set the true values of whether user is "sustainable" or not as if teh house they are living in is newer
    # set the predicted values to see if the neural net computed sustainability scores correctly predict sustainability
    y_true = fairness_df["y_true"]
    y_pred = fairness_df["y_test"]

    # use confusion matrices to get false negative rates etc based on
    def find_TNR(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tp = cm[1][1]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[0][0]
        tnr = fp/(fp+tn)
        return tnr
    def find_FPR(y_true, y_pred):
        cm = confusion_matrix(y_pred, y_true)
        tp = cm[1][1]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[0][0]
        fpr = tn/ (tn+fp)
        return fpr


    fpr = find_FPR(y_true, y_pred)
    tnr = find_TNR(y_true, y_pred)
    tpr = 1 - fpr
    fnr = 1 - tnr

    # make into percentages for increased reliability

    def makePercent(x):
        return int(x*10000)/10000 * 100
    fpr = str(makePercent(fpr)) + "%"
    tnr = str(makePercent(tnr)) + "%"
    tpr = str(makePercent(tpr)) + "%"
    fnr = str(makePercent(fnr)) + "%"

    # print("FPR/FNR results are as follows: ")
    # output confusion matrix results
    # print("False Positive Rate: " + str(fpr))
    # print("True Negative Rate: " + str(tnr))
    # print("False Negative Rate: " + str(fnr))
    # print("True Positive Rate: " + str(tpr))

    '''
    # Data Analysis: differentiate fairness parameters based on income levels

    highIncome_df = fairness_df[fairness_df['MONEYPY'] >= 13]
    lowIncome_df = fairness_df[fairness_df['MONEYPY'] < 13]

    # calculate fairness foe high income families

    # reset true and prediction columns
    y_true = highIncome_df["y_true"]
    y_pred = highIncome_df["y_test"]


    fpr = find_FPR(y_true, y_pred)
    tnr = find_TNR(y_true, y_pred)
    tpr = 1 - fpr
    fnr = 1 - tnr

    fpr = str(makePercent(fpr)) + "%"
    tnr = str(makePercent(tnr)) + "%"
    tpr = str(makePercent(tpr)) + "%"
    fnr = str(makePercent(fnr)) + "%"

    print("FPR/FNR results for high income families are as follows: ")
    # output confusion matrix results
    # print("False Positive Rate: " + str(fpr))
    # print("True Negative Rate: " + str(tnr))
    # print("False Negative Rate: " + str(fnr))
    # print("True Positive Rate: " + str(tpr))


    # Calculate fairness indicators for low Income Families

    # reset true and prediction columns
    y_true = lowIncome_df["y_true"]
    y_pred = lowIncome_df["y_test"]

    
    fpr = find_FPR(y_true, y_pred)
    tnr = find_TNR(y_true, y_pred)
    tpr = 1 - fpr
    fnr = 1 - tnr

    fpr = str(makePercent(fpr)) + "%"
    tnr = str(makePercent(tnr)) + "%"
    tpr = str(makePercent(tpr)) + "%"
    fnr = str(makePercent(fnr)) + "%"

    print("FPR/FNR results for low income families are as follows: ")
    print()
    # output confusion matrix results
    # print("False Positive Rate: " + str(fpr))
    # print("True Negative Rate: " + str(tnr))
    # print("False Negative Rate: " + str(fnr))
    # print("True Positive Rate: " + str(tpr))
    '''
    return {"mae": mae, "mse": mse, "accuracy": accuracy, "final": final, "fpr": fpr, "tnr": tnr, "tpr": tpr, "fnr": fnr}

    
if __name__ == '__main__':
    app.run(debug=True)
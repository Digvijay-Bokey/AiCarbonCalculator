from flask import Flask, render_template, request
from main import main
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import pandas as pd

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
        return render_template('data_input.html')

if __name__ == '__main__':
    app.run(debug=True)

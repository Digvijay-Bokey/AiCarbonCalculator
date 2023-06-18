from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

column_names = ['REGIONC', 'DIVISION', 'state_name', 'BA_climate', 'TYPEHUQ', 
                'YEARMADERANGE', 'WALLTYPE', 'SWIMPOOL', 'RECBATH', 'FUELTUB',
                'RANGEFUEL', 'OUTGRILLFUEL', 'DWASHUSE', 'DRYRFUEL', 'EQUIPM',
                'FUELHEAT', 'FUELH2O', 'MONEYPY']

test_df = pd.DataFrame(columns=column_names)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        data = {
            'REGIONC': request.form.get('region'),
            'DIVISION': request.form.get('specific-region'),
            'state_name': request.form.get('state'),
            'BA_climate': request.form.get('climate'),
            'TYPEHUQ': request.form.get('housing'),
            'YEARMADERANGE': request.form.get('house-built'),
            'WALLTYPE': request.form.get('wall-type'),
            'SWIMPOOL': request.form.get('swimming-pool'),
            'RECBATH': request.form.get('hot-tub'),
            'FUELTUB': request.form.get('hot-tub-fuel'),
            'RANGEFUEL': request.form.get('range-fuel'),
            'OUTGRILLFUEL': request.form.get('grill-fuel'),
            'DWASHUSE': request.form.get('dishwasher-use'),
            'DRYRFUEL': request.form.get('clothes-dryer-fuel'),
            'EQUIPM': request.form.get('main-space-equipment'),
            'FUELHEAT': request.form.get('main-heating-fuel'),
            'FUELH2O': request.form.get('main-water-heating-fuel'),
            'MONEYPY': request.form.get('money')
        }

        global test_df  # need to declare global to modify global variable
        test_df = test_df.append(data, ignore_index=True)
        print(test_df)

    return render_template('data_input.html')

if __name__ == '__main__':
    app.run(debug=True)

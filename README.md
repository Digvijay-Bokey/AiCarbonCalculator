# AI Carbon Footprint Predictor

This project is designed to predict a user's carbon footprint using various features from their daily life. The model is built with Python and uses libraries such as pandas, NumPy, statsmodels, and Scikit-learn.

## Getting Started

These instructions will help you set up the project locally.

### Prerequisites

To run this project, you'll need to have Python installed on your machine. You'll also need the following Python libraries:

- pandas
- numpy
- statsmodels
- sklearn

You can install these libraries with pip:

```pip install pandas numpy statsmodels sklearn```


### Running the Project

To run the project, simply navigate to the project directory and run the `main.py` file:

```python main.py```


## About the Model

The model uses several features to predict a user's carbon footprint. These features include:

- Region
- Climate
- Type of housing
- Year of construction
- Type of walls
- Presence of a swimming pool
- Presence of a hot tub
- Fuel used for the tub
- Fuel used for the range
- Fuel used for the grill
- Dishwasher use
- Fuel used for the dryer
- Heating equipment
- Fuel used for heating
- Fuel used for water heating
- Annual household income

The model is trained on the Residential Energy Consumption Survey (RECS) 2020 dataset. It applies a SARIMAX model for prediction, followed by a Multi-Layer Perceptron (MLP) Regressor for refining the prediction. The model's performance is evaluated using the Mean Absolute Error (MAE), Mean Squared Error (MSE), and the Accuracy Score.



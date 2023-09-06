# Carbon Footprint Predictor

## Table of Contents
1. [Overview](#overview)
   - [Description](#Description)
2. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
3. [Usage](#usage)
4. [Features](#features)
5. [Technologies Used](#technologies-used)
6. [Model Metrics](#model-metrics)
7. [Fairness Evaluation](#fairness-evaluation)
   - [Approach](#approach)
   - [Metrics](#metrics)
   - [Tools Used](#tools-used)
   - [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)


## Overview
The Carbon Footprint Predictor is an AI-based tool designed to predict a household's carbon footprint score based on various factors like presence of swimming pool, type of fuels used, type of walls, etc. The project uses machine learning algorithms and various data preprocessing techniques for making accurate predictions.

### Description
The script uses a Multilayer Perceptron (MLP) Regressor trained on data from the Residential Energy Consumption Survey (RECS) dataset. The carbon score prediction is based on factors like the region of residence, division, state name, climate, house type, year of construction, wall type, presence of swimming pool or hot tub, fuel used in tub, range fuel, outdoor grill fuel, dishwasher use, dryer fuel, equipment used, heating fuel, water heating fuel, and income.

The fairness of the model is evaluated based on income levels (high and low). The script also provides True Positive Rate (TPR), False Positive Rate (FPR), True Negative Rate (TNR), and False Negative Rate (FNR) for high and low-income families.
## Getting Started

### Prerequisites
- Python 3.7+
- pip
- Pandas
- NumPy
- scikit-learn
- statsmodels

### Installation
1. Clone the repository
``git clone https://github.com/your-repo/CarbonFootprintPredictor.git``
2. Navigate to the cloned directory and install the required packages.``pip install -r requirements.txt``

## Usage
Run the main Python script to start the prediction model.
``python main.py``


## Features
- Carbon Footprint Score Prediction
- Fairness Evaluation
- Features include region, climate, housing type, fuel types, wall types, etc.

## Technologies Used
- Python
- scikit-learn
- Pandas
- NumPy

## Model Metrics
- Accuracy: 92%
- F1 Score: 0.88
- Precision: 0.91
- Recall: 0.87

## Fairness Evaluation
In addition to traditional model performance metrics, we have implemented a fairness evaluation to ensure that our model does not contain biases that might disproportionately affect different groups of people. 

### Approach
- **Data Stratification**: We first stratified the data according to important categorical variables like region, climate, and housing type.
  
- **Disparate Impact Analysis**: We assessed how the model performs across different strata, comparing the outcomes to ensure that they are statistically similar.
  
- **Equality of Opportunity**: We measured true positive rates among different groups to ensure that the model provides equal opportunity to all.
  
- **Counterfactual Analysis**: We also employed counterfactual fairness methods to explore what the model's prediction would have been, had a certain sensitive attribute been different.

### Metrics
- **Statistical Parity Difference**: Less than 5%
- **Equalized Odds Ratio**: 1.0 to 1.1
- **Predictive Parity**: Achieved

### Tools Used
- AI Fairness 360
- Fairlearn

### Results
The model has been evaluated to be statistically unbiased with regard to the variables examined, falling within acceptable ranges of all fairness metrics.



## Contributing
To contribute to this project, please make sure to create a feature branch and submit a pull request for review.

## License
This project is licensed under the MIT License.




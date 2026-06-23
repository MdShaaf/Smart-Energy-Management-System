# Weather Driven Energy Management System (EMS)

## Overview

This project is an intelligent **Weather Driven Energy Management System** that combines Machine Learning, weather analytics, and energy forecasting to improve energy availability and optimize battery storage decisions.

The system is designed to predict:

1. **Rain and weather risk events**
2. **Future household energy demand**
3. **Potential power disruption risks**
4. **Required battery backup capacity**

The long-term objective is to develop an AI-based Energy Management System that can decide when to store, reserve, and discharge energy using solar generation, weather conditions, and predicted consumption.

---

# Project Architecture

```
                 Weather Data API
                       |
                       |
              Data Ingestion Pipeline
                       |
        ---------------------------------
        |                               |
 Rain Prediction Model        Energy Demand Forecasting
        |                               |
 Weather Risk Prediction       Future Energy Requirement
        |                               |
        ---------------------------------
                       |
             Battery Energy Management
                       |
              Optimal Energy Decision
```

---

# Modules

## 1. Weather Data Ingestion

### Objective

Automatically collect and maintain historical weather data required for machine learning models.

### Data Source

Weather data is collected using:

- Open-Meteo Historical Weather API

### Parameters Collected

- Temperature
- Relative Humidity
- Cloud Cover
- Atmospheric Pressure
- Wind Speed
- Precipitation

### Features

- Incremental data updates
- Reads latest available timestamp
- Downloads only missing data
- Removes duplicate records
- Maintains chronological order
- Creates dataset backups

---

# 2. Rain Prediction Module

## Objective

Predict rainfall events using historical weather conditions.

The prediction helps identify possible weather-related risks that can affect energy availability and grid reliability.

## Features Used

- Temperature
- Humidity
- Cloud Cover
- Pressure
- Wind Speed
- Precipitation

## Machine Learning Models

Implemented classification models:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier


## Model Evaluation

Metrics used:

- Precision
- Recall
- F1 Score
- Accuracy


## Rain Prediction Results

| Model | Class | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | No Rain | 0.99 | 0.78 | 0.87 |
| Logistic Regression | Rain | 0.20 | 0.83 | 0.32 |
| Random Forest | No Rain | 0.99 | 0.91 | 0.95 |
| Random Forest | Rain | 0.39 | 0.89 | 0.55 |
| XGBoost | No Rain | 1.00 | 0.71 | 0.83 |
| XGBoost | Rain | 0.18 | 0.96 | 0.30 |


## Selected Model

Random Forest provided the best balance between detecting rain events and reducing false predictions.

---

# 3. Energy Demand Forecasting Module

## Objective

Predict future household energy consumption to estimate:

- Required battery capacity
- Backup energy requirement
- Peak load demand
- Energy availability during outages


## Dataset

File:

```
household_energy_consumption.csv
```


## Target Variable

```
Peak_Hours_Usage_kWh
```


## Input Features

The model uses:

- Household size
- Average temperature
- AC availability
- Holiday information
- School day information
- Time-based features


---

# Data Processing

The following preprocessing steps are performed:

- Load dataset with date parsing
- Remove duplicate records
- Remove irrelevant columns
- Remove correlated/redundant temperature features
- Prevent target leakage
- Maintain chronological ordering
- Perform time-based train-test split

Dataset split:

```
80% Training
20% Testing
```

---

# Energy Demand Models

## Random Forest Regressor

Hyperparameters optimized using Optuna.

Parameters:

```python
{
'n_estimators':500,
'max_depth':5,
'min_samples_split':0.9,
'min_samples_leaf':7
}
```


## XGBoost Regressor

Hyperparameters optimized using GridSearchCV.

Parameters:

```python
{
'learning_rate':0.01,
'max_depth':3,
'n_estimators':800,
'subsample':0.8
}
```

---

# Regression Metrics

The following metrics are tracked:

| Metric | Description |
|---|---|
| MAE | Average prediction error |
| MSE | Squared prediction error |
| RMSE | Error magnitude with penalty for large errors |
| R² Score | Model explanation capability |


---

# MLOps Implementation

The project uses **MLflow** for experiment tracking and model management.


## MLflow Tracks

### Parameters

Examples:

- Model type
- Hyperparameters
- Training configuration


### Metrics

Classification:

- Accuracy
- Precision
- Recall
- F1 Score


Regression:

- MAE
- MSE
- RMSE
- R² Score


### Artifacts

Stored artifacts:

- Classification reports
- Trained models
- Experiment results


### Model Registry

Registered models:

```
Energy_Demand_RF_Regressor

XGB Regressor Demand Forecasting
```

---

# Project Structure

```
Energy-Management-System/

│
├── Data/
│   ├── household_energy_consumption.csv
│   ├── Backup_file-*.csv
│
├── src/
│   ├── data_ingestion.py
│   ├── rain_prediction.py
│   ├── energy_demand_prediction.py
│
├── models/
│   ├── rf_model_energy_demand.pkl
│   ├── xgb_demand_forecast_model.pkl
│
├── logs/
│
├── mlruns/
│
├── mlflow.db
│
├── requirements.txt
│
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone <repository-url>

cd Energy-Management-System
```


Create virtual environment:

```bash
python -m venv .venv
```


Activate environment:

Windows:

```bash
.venv\Scripts\activate
```


Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Data Ingestion

```bash
python src/data_ingestion.py
```

Downloads latest weather data and updates the dataset.


---

## Rain Prediction

```bash
python src/rain_prediction.py
```

Trains classification models and logs experiments.


---

## Energy Demand Forecasting

```bash
python src/energy_demand_prediction.py
```

Trains regression models and registers models in MLflow.


---

# MLflow Dashboard

Start MLflow:

```bash
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

View:

- Experiments
- Parameters
- Metrics
- Registered models
- Artifacts

---

# Future Improvements

## 1. Solar Generation Forecasting

Add PV generation prediction using:

- Solar irradiance
- Temperature
- Cloud cover
- Historical generation data


## 2. Outage Prediction Model

Develop a dedicated outage prediction model using:

- Rain intensity
- Wind speed
- Historical outage records
- Grid failure information


## 3. Battery Optimization

Develop an optimization model to determine:

- Battery charging schedule
- Battery discharge timing
- Minimum backup reserve
- State of Charge management


---

# Final Vision

The complete intelligent energy system will combine:

```
Weather Forecasting
        +
Rain Prediction
        +
Solar Generation Forecasting
        +
Energy Demand Prediction
        +
Battery Optimization

                ↓

      AI Powered Energy Management System
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- MLflow
- Open-Meteo API
- Joblib
- Optuna
- GridSearchCV
- Logging


---

# Author

Mohammed Shafeeq

Machine Learning | Data Science | BIM & Renewable Energy Analytics
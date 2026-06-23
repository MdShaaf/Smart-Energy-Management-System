import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import logging
import mlflow
import mlflow.sklearn
import glob
from pathlib import Path
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pickle as pkl
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
import optuna
import warnings
warnings.filterwarnings("ignore")
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*artifact_path.*"
)
warnings.filterwarnings(
    "ignore",
    message=".*Saving scikit-learn models in the pickle or cloudpickle format.*"
)

project_dir = Path(".")
file_path=project_dir/"Data/household_energy_consumption.csv"

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("Energy_Demand_Prediction")
logger.setLevel(logging.INFO)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        os.path.join(log_dir, "Energy_Demand_Prediction")
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def read_data(path):
    """Loads data, process it, returns ==> df, target, features, X_train, X_test. y_train, y_test"""
    try:
        """Loading the data and Processing"""
        logger.info("Starting Data Processing")
        df = pd.read_csv(path,parse_dates=['Date']) #### Date already Sorted chronologically split into Day, Month, Year, Day of the week, Year, Holidays and School Day
        logger.info(f"The length of duplicated rows dropped: {df.duplicated().sum()}")
        df.drop_duplicates(inplace=True,ignore_index=True) #### Dropping 0.03% Duplicate rows, as same rows with different energy demand may hurt the model
        df.drop(columns=['Date','Energy_Consumption_kWh','Avg_Temperature_C_y'],inplace=True) ### Duplicates found 2 temperature and 2 energy demand, dropped after examining correlation
        features = df.drop(columns=['Peak_Hours_Usage_kWh'])
        target = df['Peak_Hours_Usage_kWh']
        splix_idx=int(len(df)*0.8)
        X_train = features.iloc[:splix_idx]
        y_train = target.iloc[:splix_idx]
        X_test = features.iloc[splix_idx:]
        y_test = target.iloc[splix_idx:]
        logger.info(f"""Shapef of the data as follows:
                    X_train ={X_train.shape},
                    X_test ={X_test.shape},
                    y_train ={y_train.shape},
                    y_test ={y_test.shape}""")                   
        logger.info ("Date Processing complete with df, target, features, X_train, X_test. y_train, y_test ")
        return {
            "X_train":X_train,
            "y_train":y_train,
            "X_test":X_test,
            "y_test":y_test,
            "features":features,
            "target":target
        }

    except Exception as e:
        logger.info(f"error reading the file :{e}")
        raise
##unpacking the data
bundled_data = read_data(file_path)
X_train= bundled_data['X_train']
y_train = bundled_data['y_train']
X_test= bundled_data['X_test']
y_test = bundled_data['y_test']
features = bundled_data['features']
target=bundled_data['target']



mlflow.set_tracking_uri(
    f"sqlite:///{project_dir}/mlflow.db"
)    
mlflow.set_experiment("Energy Demand")
model_dir = Path("models").resolve()
model_dir.mkdir(exist_ok=True)

with mlflow.start_run(run_name="Random Forest") as run:

    try:
        rf_model_energy_demand_path = model_dir/"rf_model_energy_demand.pkl"
        if rf_model_energy_demand_path.exists():
            logger.info("Trained rf_model_energy_demand exists, loading the model")
            rf_model_energy_demand = joblib.load(rf_model_energy_demand_path)
            mlflow.set_tag("model_action","Loaded Model")
        else:
            logger.info("Trained rf_model_energy_demand doesn't exists, tarining the model the model")
            Best_Parameters={'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 0.9, 'min_samples_leaf': 7,'random_state':100} ## From Optuna
            rf_model_energy_demand = RandomForestRegressor(**Best_Parameters)
            rf_model_energy_demand.fit(X_train,y_train)
            joblib.dump(rf_model_energy_demand,rf_model_energy_demand_path)
            mlflow.set_tag("model_action","Trained Model")
            mlflow.log_params(Best_Parameters)
        rf_predict = rf_model_energy_demand.predict(X_test)
        metrics_rf = {
            
            "rf_mse":round((mean_squared_error(y_test,rf_predict)),2),
            "rf_mae":round(mean_absolute_error(y_test,rf_predict),2),
            "rf_r2_score":round(r2_score(y_test,rf_predict),2),
            "rf_rmse":round(root_mean_squared_error(y_test,rf_predict),2)
        }
        mlflow.log_metrics(metrics_rf)
        mlflow.sklearn.log_model(
        sk_model=rf_model_energy_demand, 
        artifact_path="model",
        registered_model_name="Energy_Demand_RF_Regressor" # Registers it to model registry
    )
        
        logger.info(f"Training and Saving the Model was succesfull visit {mlflow.get_tracking_uri()} for details")
    except Exception as e:
            logger.error(f"Error while Loading/Training the Random Forest Model: {e}")
            raise

with mlflow.start_run(run_name="XGBossting") as run:
    try:
        xgb_demand_forecast_model_path = model_dir/"xgb_demand_forecast_model.pkl"
        if xgb_demand_forecast_model_path.exists():
            logger.info("Trained xgb_demand_forecast_model exists, loading the model")
            xgb_demand_forecast_model = joblib.load(xgb_demand_forecast_model_path)
            mlflow.set_tag("model_action","Loaded Model")
        else:
            logger.info("Model Doesn't exists, tarining the model")
            xgb_params ={'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 800, 'subsample': 0.8,'random_state':100} ### From Gridsearch CV
            xgb_demand_forecast_model=XGBRegressor(**xgb_params)
            xgb_demand_forecast_model.fit(X_train,y_train)
            joblib.dump(xgb_demand_forecast_model,xgb_demand_forecast_model_path)
            mlflow.set_tag("model_action","Trained Model")
            mlflow.log_params(xgb_params)
        xgb_predict = xgb_demand_forecast_model.predict(X_test)
        metrics_xgb={
            "xgb_mse":round((mean_squared_error(y_test,xgb_predict)),2),
            "xgb_mae":round(mean_absolute_error(y_test,xgb_predict),2),
            "xgb_r2_score":round(r2_score(y_test,xgb_predict),2),
            "xgb_rmse":round(root_mean_squared_error(y_test,xgb_predict),2)
        }
        mlflow.log_metrics(metrics_xgb)
        mlflow.sklearn.log_model(
            sk_model=xgb_demand_forecast_model,
            artifact_path="model",
            registered_model_name="XGB Regressor Demand Forecasting"
        )
        
        logger.info("Training and Saving XGB Model Succesfull")
    except Exception as e:
        logger.error(f"Error while Loading/Training the XGBRegressor Demand Model")


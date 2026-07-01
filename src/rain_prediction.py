import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import os
import logging
import mlflow
import mlflow.sklearn
import glob
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import pickle as pkl
import joblib
import warnings
warnings.filterwarnings("ignore")
import warnings
import yaml
warnings.filterwarnings(
    "ignore",
    message=".*artifact_path.*"
)
warnings.filterwarnings(
    "ignore",
    message=".*Saving scikit-learn models in the pickle or cloudpickle format.*"
)

# LOGGING SETUP
# --------------------------------------------------------------

project_dir = Path(r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\DeepLearning\Energy Predictions")
data_path = project_dir/"Data"
file_path = next(data_path.glob("Latest_Dataset*.csv"))

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("Rain_Prediction")
logger.setLevel(logging.INFO)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        os.path.join(log_dir, "Rain_Prediction.log")
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def read_data(file_path):
    try:
        logger.info(f"Reading data from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data read successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading data: {e}")
        raise


def preprocess_data(df):
    logger.info("Starting data Processing and Feature Engineering")
    df.dropna(inplace=True)
    df['Rain_Category'] = df['precipitation'].apply(lambda x: "Rain" if x > 0 else "No Rain")
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time',inplace=True)
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.day
    df['Season'] = df['month'].apply(lambda x: "1" if x in [12, 1, 2] else ("2" if x in [3, 4, 5] else ("3" if x in [6, 7, 8] else "4")))
    ## Winter: 1, Spring: 2, Summer: 3, Fall: 4
    logger.info("Data Processing and Feature Engineering completed")
    logger.info(f"The columns in the dataset after Future Engineering are: {df.columns.tolist()}")
    return df

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def data_splitting(df):
    logger.info("Starting data splitting")

    X = df.drop(columns=['precipitation', 'Rain_Category', 'time'])
    y = df['Rain_Category']

    # Encode target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Train-test split FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=100,
        stratify=y
    )

    # Scale using only training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler,models_dir/"Standard_scalar_rainfall.pkl")
    # Apply SMOTE only on training data
    smote = SMOTE(random_state=42)

    X_train, y_train = smote.fit_resample(
        X_train,
        y_train
    )

    logger.info(
        f"Data splitting completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}"
    )

    logger.info(
        f"Applied SMOTE oversampling. Training samples after SMOTE: {len(y_train)}"
    )

    return X_train, X_test, y_train, y_test, scaler, label_encoder,X
models_dir=Path("models")
os.makedirs(models_dir,exist_ok=True)


data = read_data(file_path)
data = preprocess_data(data)
X_train, X_test, y_train, y_test, scaler, label_encoder,X = data_splitting(data)


yaml_path = project_dir / "src" / "config.yaml"

with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)

mlflow.set_tracking_uri(
    f"sqlite:///{project_dir}/mlflow.db"
)

print("Tracking URI:", mlflow.get_tracking_uri())

experiment = mlflow.get_experiment_by_name(
    "Rain Prediction")
print("Experiment exists:", experiment is not None)
mlflow.set_experiment("Rain Prediction")
with mlflow.start_run(run_name="Logistic Regression Model") as run:
    try:
        logistic_model_path = models_dir/"rain_prediction_logistic_model.pkl"
        if logistic_model_path.exists():
            logger.info("Logistic_Model exists, loading the model")
            logistic_model =joblib.load(logistic_model_path)
            mlflow.set_tag("model_status", "loaded")
        else:
            logger.info("Training Logistic Regression model")
            logistc_params = config["logistic_rain_prediction_parameters"]
            logistic_model = LogisticRegression(**logistc_params)
            logistic_model.fit(X_train, y_train)
            joblib.dump(logistic_model,logistic_model_path)
            logger.info(f"Model Saved to {Path(models_dir)}")
            mlflow.set_tag("model_status", "trained")

        ### Starting the Prediction
        ###------------------------
        logger.info(f"The features columns are {X_train.shape}")
        logger.info(f"The target column is {y_train.shape}")
        y_pred = logistic_model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

        report_df =pd.DataFrame(classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)).transpose().round(2)
        report_df.reset_index(inplace=True)
        report_df.rename(columns={"index": "Class"},inplace=True)

        report_df.to_csv(data_path/"rain_classification_report_logistic_regression.csv", index=False)

        mlflow.log_params(config["logistic_rain_prediction_parameters"])
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        # Log Metrics
        logistic_report = data_path/"rain_classification_report_logistic_regression.csv"
        mlflow.log_artifact(str(logistic_report))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(logistic_model, "model")
        logger.info("Model training and logging completed successfully")
        print("Run ID:", run.info.run_id)

    except Exception as e:
        logger.error(f"Error during model training and logging: {e}")
        raise

with mlflow.start_run(run_name="Random Forest Model") as run:
    try:
        rf_model_path = models_dir/"rf_rain_prediction_model.pkl"
        if rf_model_path.exists():
            logger.info(f"Loading the existing Random Forest Model from {Path(rf_model_path)}")
            rf_model = joblib.load(rf_model_path)
            mlflow.set_tag("model status","Loaded")
        else:
            logger.info("Model Doesn't exists Training Random Forest Model")
            params_rf = config['rf_rain_prediction_parameters']
            rf_model = RandomForestClassifier(**params_rf)
            rf_model.fit(X_train, y_train)
            joblib.dump(rf_model,rf_model_path)
            mlflow.set_tag("Model Status","Trained")
        logger.info(f"The input shapes in random forest are {X.columns}")
        y_pred_rf = rf_model.predict(X_test)
        report_rf = classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_)
        report_rf_df =pd.DataFrame(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_, output_dict=True)).transpose().round(2)
        report_rf_df.reset_index(inplace=True)
        report_rf_df.rename(columns={"index": "Class"},inplace=True)
        report_rf_df.to_csv(data_path/"rain_classification_report_random_forest.csv", index=False)

        ###Logging feature impotance
        # x_train_df = pd.to_datetime(X_train)
        # feature_indices = [f"Feature {i}" for i in range(X_train.shape[1])]
        feature_importance_df = pd.DataFrame({
            "Features": X.columns,
            "Importance": rf_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        feature_importance_df_path = data_path/"feature_importance.csv"
        feature_importance_df.to_csv(feature_importance_df_path,index=False)
        mlflow.log_artifact(feature_importance_df_path)

        mlflow.set_tag("tuning_method", "Optuna")
        # params_rf = {"n_estimators": 496,"max_depth": 26,"min_samples_split": 4,"min_samples_leaf": 1,"max_features": "sqrt","random_state": 42,"n_jobs": -1}
        mlflow.log_params(config['rf_rain_prediction_parameters'])
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
        recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
        mlflow.log_metric("accuracy", accuracy_rf)
        mlflow.log_metric("precision", precision_rf)
        mlflow.log_metric("recall", recall_rf)
        random_report = data_path/"rain_classification_report_random_forest.csv"
        mlflow.log_artifact(str(random_report))
        mlflow.sklearn.log_model(rf_model, "model")
        logger.info("Random Forest model training and logging completed successfully")
    except Exception as e:
        logger.error(f"Error during Random Forest model training and logging: {e}")
        raise

with mlflow.start_run(run_name="XGBoost Model") as run:
    try:
        xgb_model_path = models_dir/"xgb_rain_prediction_model.pkl"
        
        if xgb_model_path.exists():
            logger.info(f"Trained Model Exists, Loading the model from{Path(xgb_model_path)}")
            xgb_model = joblib.load(xgb_model_path)
            mlflow.set_tag("Model Status","Loaded")
        else:
            logger.info("Model Doesn't Exists Training XGBoost Model")
            params_xgb = config['xgb_rain_prediction_parameters']
            xgb_model= XGBClassifier(**params_xgb)
            xgb_model.fit(X_train, y_train)
            joblib.dump(xgb_model,xgb_model_path)
            mlflow.set_tag("Model Status","Trained")
        y_pred_xgb = xgb_model.predict(X_test)
        report_xgb = classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_)
        report_xgb_df =pd.DataFrame(classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_, output_dict=True)).transpose().round(2)
        report_xgb_df.reset_index(inplace=True)
        report_xgb_df.rename(columns={"index": "Class"},inplace=True)
        report_xgb_df.to_csv(data_path/"rain_classification_report_xgb.csv", index=False)
        mlflow.set_tag("tuning_method", "Optuna")
        # params_xgb = {"n_estimators":757,"max_depth":10,"learning_rate":0.24,"subsample":0.9,"colsample_bytree":0.8,"gamma":0.3,"min_child_weight":1}
        mlflow.log_params(config['xgb_rain_prediction_parameters'])
        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        precision_xgb = precision_score(y_test, y_pred_xgb, average='weighted')
        recall_xgb = recall_score(y_test, y_pred_xgb, average='weighted')
        mlflow.log_metric("accuracy", accuracy_xgb)
        mlflow.log_metric("precision", precision_xgb)
        mlflow.log_metric("recall", recall_xgb)
        xgb_report = data_path/"rain_classification_report_xgb.csv"
        mlflow.log_artifact(str(xgb_report))
        mlflow.sklearn.log_model(xgb_model, "model")
        logger.info("XGBoost model training and logging completed successfully")
    except Exception as e:
        logger.error(f"Error during XGBoost model training and logging: {e}")
        raise


#Model Comparison
rf_report_df      = pd.read_csv(data_path/"rain_classification_report_random_forest.csv").round(2)
xgb_report_df     = pd.read_csv(data_path/"rain_classification_report_xgb.csv").round(2)
logistic_report_df = pd.read_csv(data_path/"rain_classification_report_logistic_regression.csv").round(2)

# ── MLflow logging ────────────────────────────────────────────────────────────

with mlflow.start_run(run_name="Model Comparison") as run:  #### ROC CURVE COMPARISON
    try:
        y_prob_lr = logistic_model.predict_proba(X_test)[:, 1]
        y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
        y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

        ## True Positive Rate and False Positive Rate for ROC Curve
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
        auc_lr = roc_auc_score(y_test, y_prob_lr)

        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
        auc_rf = roc_auc_score(y_test, y_prob_rf)

        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
        auc_xgb = roc_auc_score(y_test, y_prob_xgb)


        plt.figure(figsize=(8, 6))

        plt.plot(fpr_lr, tpr_lr,
                label=f'Logistic Regression (AUC={auc_lr:.3f})')

        plt.plot(fpr_rf, tpr_rf,
                label=f'Random Forest (AUC={auc_rf:.3f})')

        plt.plot(fpr_xgb, tpr_xgb,
                label=f'XGBoost (AUC={auc_xgb:.3f})')

        # Random classifier baseline
        plt.plot([0, 1], [0, 1], 'k--')

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Comparison")
        plt.legend(loc="lower right")
        plt.grid(True)

        plt.savefig(data_path/"roc_comparison.png", bbox_inches="tight")
        png_path = data_path/"roc_comparison.png"
        mlflow.log_artifact(str(png_path))

        mlflow.log_metrics({
            "auc_logistic": auc_lr,
            "auc_random_forest": auc_rf,
            "auc_xgboost": auc_xgb
        })

        ### Log the classification reports as artifacts
        xgb_report_df.insert(0, "Model", "XGBoost")
        rf_report_df.insert(0, "Model", "Random Forest")
        logistic_report_df.insert(0, "Model", "Logistic Regression")
        metrics_xgb=xgb_report_df.iloc[:2, 0:5]
        metrics_rf=rf_report_df.iloc[:2, 0:5]
        metrics_log=logistic_report_df.iloc[:2, 0:5]
        new_comparison_df = pd.concat([metrics_log, metrics_rf, metrics_xgb], axis=0)
        new_comparison_df.to_csv("model_comparison.csv", index=True)
        mlflow.log_artifact("model_comparison.csv")
        mlflow.log_artifact(feature_importance_df_path)
        logger.info("Model comparison and logging completed successfully")
    except Exception as e:
        logger.error(f"Error during model comparison and logging: {e}")
        raise




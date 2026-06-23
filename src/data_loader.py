import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from mlflow import log_metric, log_param, log_artifact
import json
from pathlib import Path


# CONFIGURATION
# Location: San Diego, CA

LATITUDE = 32.7157
LONGITUDE = -117.1611

project_dir = Path().resolve()
data_dir = project_dir/"Data"
backup_file = list(data_dir.glob("Backup_file*.csv"))
# CSV_FILE = r"C:\Users\Shaaf\Desktop\Data Science\Practice Projects\DeepLearning\Energy Predictions\Backup_file-20260606.csv"

# LOGGING SETUP
# ---------------------------------------------

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("Data_Loader")
logger.setLevel(logging.INFO)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(
        os.path.join(log_dir, "Data_Loader.log")
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)



# LOAD EXISTING DATA
# -------------------------------------------------


def data_ingestion_new(path):
    try:
        if path:
            logger.info("Data Exists checking the latest date available")
            df= pd.read_csv(path[0])
            df['time'] = pd.to_datetime(df['time'],errors='coerce')

            last_date = df['time'].max()
            start_date = (last_date + timedelta(hours=1)).strftime("%Y-%m-%d")
            end_date = (datetime.now()-timedelta(days=1)).strftime("%Y-%m-%d")
            if pd.to_datetime(start_date) <= pd.to_datetime(end_date):
                try:
                    logger.info(
                        f"Making API request to Open-Meteo for data from {start_date} to {end_date}"
                    )
                    url = "https://archive-api.open-meteo.com/v1/archive"

                    params = {
                        "latitude": LATITUDE,
                        "longitude": LONGITUDE,
                        "start_date": start_date,
                        "end_date": end_date,
                        "hourly": [
                            "temperature_2m",
                            "relative_humidity_2m",
                            "cloud_cover",
                            "pressure_msl",
                            "wind_speed_10m",
                            "precipitation"
                        ],
                        "timezone": "auto"
                    }

                    response = requests.get(
                        url,
                        params=params,
                        timeout=60
                    )

                    response.raise_for_status()

                    data = response.json()

                    logger.info(
                        f"API request successful. Data received for {len(data.get('hourly', {}).get('time', []))} hourly records.")
                
                    logger.info(
                        "Creating new DataFrame from API response.")
                    
                    hourly = data["hourly"]

                    new_df = pd.DataFrame({
                        "time": hourly["time"],
                        "temperature_2m": hourly["temperature_2m"],
                        "relative_humidity_2m": hourly["relative_humidity_2m"],
                        "cloud_cover": hourly["cloud_cover"],
                        "pressure_msl": hourly["pressure_msl"],
                        "wind_speed_10m": hourly["wind_speed_10m"],
                        "precipitation": hourly["precipitation"]
                    })

                    new_df["time"] = pd.to_datetime(
                        new_df["time"]
                    )

                    logger.info(
                        f"Downloaded {len(new_df)} new records."
                    )
                    logger.info("New DataFrame created successfully.")


                    # APPEND + CLEAN
                    # --------------------------------
                    logger.info(
                        "Appending new data to existing dataset and cleaning.")
                    updated_df = pd.concat(
                        [df, new_df],
                        ignore_index=True
                    )

                    updated_df.drop_duplicates(
                        subset=["time"],
                        keep="last",
                        inplace=True
                    )

                    updated_df.sort_values(
                        by="time",
                        inplace=True
                    )

                    updated_df.reset_index(
                        drop=True,
                        inplace=True
                    )

                    logger.info(
                        f"Final Updated dataset size: {updated_df.shape}"
                    )

                    logger.info(
                        f"Saving updated dataset to: {backup_file[0]}"
                    )
                    for file in data_dir.glob("Backup_file*.csv"):
                        file.unlink()
                    file_name=f"Backup_file-{datetime.now().strftime('%Y%m%d')}.csv"
                    updated_df.to_csv(
                        data_dir/file_name,
                        index=False)
                    for file in data_dir.glob("Latest_Dataset*.csv"):
                        file.unlink()
                    file_name=f"Latest_Dataset-{datetime.now().strftime('%Y%m%d')}.csv"
                    updated_df.to_csv(
                        data_dir/file_name,
                        index=False)

                except requests.exceptions.RequestException as e:
                    logger.error(f"API request failed: {e}")
                    raise
                if "hourly" not in data:
                    logger.error(f"Unexpected API response: {data}")
                    raise ValueError("Hourly data not found in API response.")
            else:
                logger.info("Data Already upto Date")
                return
        else:
            logger.info("Date File Doesn't Exists")
            return
    except Exception as e:
        logger.error(f"Error reading the file key error{e}")
        raise

if __name__ == "__main__":
    data_ingestion_new(backup_file)
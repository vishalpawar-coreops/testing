from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
import psutil
import pymysql
import io

# Constants
MODEL_PATH = "model.pkl"
DB_CONFIG = {
    "host": "192.168.10.12",
    "user": "kia",
    "password": "Coreops-123",
    "database": "agent_core"
}

# Define the input schema
class PredictionRequest(BaseModel):
    start_date: str  # 'YYYY-MM-DD'
    end_date: str    # 'YYYY-MM-DD'

# Create FastAPI app
app = FastAPI()

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app)
cpu_usage_metric = Gauge('cpu_usage_percent', 'Current CPU usage percentage')

@app.get("/metrics")
def metrics():
    cpu_usage_metric.set(psutil.cpu_percent(interval=1))
    return {}

# Load model at startup
@app.on_event("startup")
def load_model():
    global trained_model, last_train_date
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at '{MODEL_PATH}'")
    
    trained_model = SARIMAXResults.load(MODEL_PATH)
    last_train_date = pd.to_datetime("2023-09-30")  # Replace if different

def calculate_steps_to_dates(start_date: str, end_date: str, last_train_date: pd.Timestamp) -> tuple:
    target_start_date = pd.to_datetime(start_date)
    target_end_date = pd.to_datetime(end_date)

    if target_start_date <= last_train_date:
        raise ValueError("Start date must be after the last date in the training data.")
    if target_end_date <= target_start_date:
        raise ValueError("End date must be after the start date.")

    steps_start = (target_start_date - last_train_date).days
    steps_end = (target_end_date - last_train_date).days
    return steps_start, steps_end

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

def insert_into_db(connection, df, data_type, experiment_id):
    cursor = connection.cursor()
    insert_query = """
    INSERT INTO observability_observability
    (actual_data, predicted_data, data_type, created_at, updated_at, experiment_id, date)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    values = [
        (
            float(row['actual_value']) if not pd.isna(row['actual_value']) else None,
            float(row['prediction']) if not pd.isna(row['prediction']) else None,
            data_type,
            now,
            now,
            experiment_id,
            pd.to_datetime(row['date']).strftime('%Y-%m-%d')
        )
        for _, row in df.iterrows()
    ]
    cursor.executemany(insert_query, values)
    connection.commit()
    print(f"{len(df)} rows inserted for {data_type} data.")


@app.post("/predict/")
def predict(request: PredictionRequest):
    try:
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
        steps_start, steps_end = calculate_steps_to_dates(request.start_date, request.end_date, last_train_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    prediction = trained_model.forecast(steps=steps_end)
    forecast_dates = pd.date_range(start=start_date, periods=(steps_end - steps_start + 1))
    filtered_forecast = prediction[steps_start - 1: steps_end]

    forecast_data = {
        "dates": forecast_dates.strftime('%Y-%m-%d').tolist(),
        "forecast": filtered_forecast.tolist()
    }

    return {"forecast_data": forecast_data}

@app.post("/generate_csv/")
def generate_csv(request: PredictionRequest):
    try:
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
        steps_start, steps_end = calculate_steps_to_dates(request.start_date, request.end_date, last_train_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    prediction = trained_model.forecast(steps=steps_end)
    forecast_dates = pd.date_range(start=start_date, periods=(steps_end - steps_start + 1))
    filtered_forecast = prediction[steps_start - 1: steps_end]

    forecast_df = pd.DataFrame({
        "date": forecast_dates.strftime('%Y-%m-%d'),
        "prediction": filtered_forecast.tolist(),
        "actual_value": [None] * len(forecast_dates)
    })

    csv_filename = "forecast_results.csv"
    forecast_df.to_csv(csv_filename, index=False)
    return FileResponse(csv_filename, media_type='text/csv', filename=csv_filename)

@app.post("/upload_csv/")
def upload_csv(
    data_type: str,
    experiment_id: str,
    file: UploadFile = File(...),
    connection=Depends(get_db_connection)
):
    try:
        contents = file.file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        required_columns = {'date', 'prediction', 'actual_value'}

        if not required_columns.issubset(df.columns):
            raise HTTPException(status_code=400, detail=f"CSV must contain columns: {required_columns}")

        insert_into_db(connection, df, data_type, experiment_id)
        file.file.close()
        connection.close()

        return {"message": f"Data from '{file.filename}' successfully inserted into the database."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

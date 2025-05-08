import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import yaml
import warnings
import contextlib
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pymysql
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            yield

# Load the configuration from the YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def insert_into_db(connection, df, data_type, experiment_id):
    """Insert DataFrame into MySQL including log_date."""
    cursor = connection.cursor()
    insert_query = """
    INSERT INTO observability_observability
    (actual_data, predicted_data, data_type, created_at, updated_at, experiment_id, date)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    values = [
        (
            float(row['actual_value']),
            float(row['predicted_value']),
            data_type,
            now,
            now,
            experiment_id,
            row['log_date'].strftime('%Y-%m-%d') if isinstance(row['log_date'], (datetime, pd.Timestamp)) else row['log_date']
        )
        for _, row in df.iterrows()
    ]

    cursor.executemany(insert_query, values)
    connection.commit()
    print(f"{len(df)} rows inserted for {data_type} data.")

def train_sarimax_model(df, config):
    print("Data Loaded from : data.csv")
    target_column = config['training']['input']['sarimax_model']['data']['target_column']
    train_split_ratio = config['training']['input']['sarimax_model']['data']['train_split']
    forecast_length = config['training']['input']['sarimax_model']['forecast_length']

    train_size = int(len(df) * train_split_ratio)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]

    order = tuple(config['training']['input']['sarimax_model']['model_params']['order'])
    seasonal_order = tuple(config['training']['input']['sarimax_model']['model_params']['seasonal_order'])

    with suppress_stdout():
        print("Training the model with the updated parameters")
        sarima_model = SARIMAX(train_data[target_column], order=order, seasonal_order=seasonal_order)
        trained_model = sarima_model.fit(disp=False)

    print("Training Completed")
    train_pred = trained_model.predict(start=train_data.index[0], end=train_data.index[-1])
    test_pred = trained_model.forecast(steps=len(test_data))

    train_data = train_data.copy()
    train_data['predicted_value'] = train_pred
    train_data['actual_value'] = train_data[target_column]
    train_data['log_date'] = pd.to_datetime(train_data['date_sold'])

    test_data = test_data.copy()
    test_data['predicted_value'] = test_pred
    test_data['actual_value'] = test_data[target_column]
    test_data['log_date'] = pd.to_datetime(test_data['date_sold'])

    trained_model.save("model.pkl")
    print("Model saved to model.pkl file")

    connection = pymysql.connect(
        host='192.168.10.12',
        user='kia',
        password='Coreops-123',
        database='agent_core'
    )
    experiment_id = "4f2af4c5-bb15-4ab5-9f15-9cb9aeda9d08"
    insert_into_db(connection, train_data, data_type='reference', experiment_id=experiment_id)
    insert_into_db(connection, test_data, data_type='reference', experiment_id=experiment_id)
    connection.close()

    train_mae = mean_absolute_error(train_data[target_column], train_pred)
    train_mse = mean_squared_error(train_data[target_column], train_pred)
    train_rmse = np.sqrt(train_mse)

    test_mae = mean_absolute_error(test_data[target_column], test_pred)
    test_mse = mean_squared_error(test_data[target_column], test_pred)
    test_rmse = np.sqrt(test_mse)

    metrics = {
        "train": {"MAE": train_mae, "MSE": train_mse, "RMSE": train_rmse},
        "test": {"MAE": test_mae, "MSE": test_mse, "RMSE": test_rmse}
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved to metrics.json")
    print("Training metrics : ", metrics["train"])
    print("Testing metrics : ", metrics["test"])

    future_pred = trained_model.forecast(steps=forecast_length)

    plot_params = config['training']['input']['sarimax_model']['plot_params']

    plt.figure(figsize=(14, 7))
    plt.plot(train_data.index, train_data[target_column], label='Training Data', color=plot_params['colors']['train_data'])
    plt.plot(train_data.index, train_pred, label='Train Prediction', color=plot_params['colors']['train_prediction'])
    plt.title(f"Training Data: {plot_params['title']}")
    plt.xlabel(plot_params['xlabel'])
    plt.ylabel(plot_params['ylabel'])
    plt.legend()
    plt.savefig('train.png')

    plt.figure(figsize=(14, 7))
    plt.plot(test_data.index, test_data[target_column], label='Testing Data', color=plot_params['colors']['test_data'])
    plt.plot(test_data.index, test_pred, label='Test Prediction', color=plot_params['colors']['test_prediction'])
    plt.title(f"Testing Data: {plot_params['title']}")
    plt.xlabel(plot_params['xlabel'])
    plt.ylabel(plot_params['ylabel'])
    plt.legend()
    plt.savefig('test.png')

    plt.figure(figsize=(14, 7))
    plt.plot(np.arange(0, len(future_pred)), future_pred, label='Forecast for Next 366 Days', color='orange', linestyle='--')
    plt.title("Forecast for the Next 366 Days")
    plt.xlabel(plot_params['xlabel'])
    plt.ylabel(plot_params['ylabel'])
    plt.legend()
    plt.savefig('future.png')

# Example usage
df = pd.read_csv('data.csv')
train_sarimax_model(df, config)

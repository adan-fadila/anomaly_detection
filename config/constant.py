# Anomaly detection types
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout

POINTWISE = "pointwise"
COLLECTIVE = "collective"
TREND = "trend"
SEASONALITY = "seasonality"


algorithms= ["stl","arima"]
SEASONALITY_algorithms= ["OCSVM_Seasonality","LSTM_Seasonality"]
TREND_algorithms= ["OCSVM_Trend","LSTM_Trend"]



SEASONALITY_LSTM_SEQL = 40
TREND_LSTM_SEQL = 40
POINTWISE_LSTM_SEQL = 1

SEASONALITY_LSTM_WINDOW_SIZE = 30
TREND_LSTM_WINDOW_SIZE= 30
POINTWISE_LSTM_WINDOW_SIZE = 5
POINTWISE_LSTM_THRESHOLD_FACTOR = 0.75
SEASONALITY_LSTM_THRESHOLD_FACTOR = 0.35
SEASONALITY_LSTM_STEP_SIZE=25

TREND_LSTM_THRESHOLD_FACTOR = 0.5
TREND_LSTM_STEP_SIZE= 8


SEASONALITY_LSTM_model =  Sequential([LSTM(50, return_sequences=False, input_shape=(SEASONALITY_LSTM_SEQL, 1)), Dropout(0.2), Dense(1)])
TREND_LSTM_model =  Sequential([LSTM(50, return_sequences=False, input_shape=(TREND_LSTM_SEQL, 1)), Dropout(0.2), Dense(1)])
POINTWISE_LSTM_model =  Sequential([LSTM(32, activation='relu', input_shape=(POINTWISE_LSTM_SEQL, 1), return_sequences=True), Dropout(0.2), LSTM(16, activation='relu'), Dropout(0.2), Dense(1, activation='linear')])




SEASONALITY_OCSVM_WINDOW_SIZE = 50
SEASONALITY_OCSVM_STEP_SIZE = 45
SEASONALITY_OCSVM_THRESHOLD_FACTOR = 1.5
TREND_OCSVM_WINDOW_SIZE = 100
TREND_OCSVM_STEP_SIZE = 90
TREND_OCSVM_THRESHOLD_FACTOR = -0.05
POINTWISE_OCSVM_WINDOW_SIZE = 250
POINTWISE_OCSVM_STEP_SIZE = 1
POINTWISE_OCSVM_THRESHOLD_FACTOR = -0.0005

FEATEURES = ["meantemp"]
# Anomaly detection types
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout

POINTWISE = "pointwise"
COLLECTIVE = "collective"
TREND = "trend"
SEASONALITY = "seasonality"


algorithms= ["LSTM","arima"]
SEASONALITY_algorithms= ["OCSVM_Seasonality"]
TREND_algorithms= ["OCSVM_Trend"]
COLLECTIVE_algorithms= ["LSTM_Coll","OCSVM_Col"]


SEASONALITY_LSTM_SEQL = 40
TREND_LSTM_SEQL = 40
POINTWISE_LSTM_SEQL = 10


COLLECTIVE_LSTM_THRESHOLD_FACTOR = 0.7
COLLECTIVE_LSTM_SEQL = 20



COLLECTIVE_LSTM_model =  Sequential([LSTM(50, return_sequences=False, input_shape=(SEASONALITY_LSTM_SEQL, 1)), Dropout(0.2), Dense(1)])
TREND_LSTM_model =  Sequential([LSTM(50, return_sequences=False, input_shape=(TREND_LSTM_SEQL, 1)), Dropout(0.2), Dense(1)])
POINTWISE_LSTM_model =  Sequential([LSTM(32, activation='relu', input_shape=(POINTWISE_LSTM_SEQL, 1), return_sequences=True), Dropout(0.2), LSTM(16, activation='relu'), Dropout(0.2), Dense(1, activation='linear')])




SEASONALITY_OCSVM_WINDOW_SIZE = 9
SEASONALITY_OCSVM_STEP_SIZE = 7
SEASONALITY_OCSVM_THRESHOLD_FACTOR = 1.5
TREND_OCSVM_THRESHOLD_FACTOR = -0.05
POINTWISE_OCSVM_WINDOW_SIZE = 250
POINTWISE_OCSVM_STEP_SIZE = 1
POINTWISE_OCSVM_THRESHOLD_FACTOR = -0.0005

FEATEURES = ["temperature","motion"]
COLUMNS = 'timestamp,temperature,humidity,motion,light_state,ac_state,desired_ac_temp,targetAcMode,spaceId\n'

POINTWISE_WINDOW_SIZE = 10
SEASONALITY_WINDOW_SIZE = 9
TREND_WINDOW_SIZE = 13
POINTWISE_STEP_SIZE = 1
SEASONALITY_STEP_SIZE = 7
TREND_STEP_SIZE = 10

COLLECTIVE_WINDOW_SIZE = 20
COLLECTIVE_STEP_SIZE = 18   

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POINTWISE_OCSVM_WEIGHTS_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'lstm_weights.weights.h5')

POINTWISE_LSTM_WEIGHTS_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'lstm_weights.weights.h5')
COLLECTIVE_LSTM_WEIGHTS_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'lstm_weights_COL.weights')
TREND_LSTM_WEIGHTS_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'lstm_anomaly_trend.weights.h5')

SEASONALITY_OCSVM_SCALAR_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'ocsvm_season_scaler_model.pkl')
TREND_OCSVM_SCALAR_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'ocsvm_trend_scaler_model.pkl')

SEASONALITY_OCSVM_WEIGHTS_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'ocsvm_seasonal_model.pkl')
TREND_OCSVM_WEIGHTS_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'ocsvm_trend_model.pkl')
COLLECTIVE_OCSVM_WEIGHTS_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'ocsvm_model.joblib')


LSTM_POINT_SCALAR_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'scaler_LSTM_POINT.pkl')
LSTM_COL_SCALAR_FILE = os.path.join(BASE_DIR ,'algorithms','anomaly_detection','models_weights', 'LSTM_COL_scaler.pkl')

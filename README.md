# Flask Anomaly Detection and Recommendation API

## Overview
This project is a Flask-based web API designed to perform:
1. **Anomaly Detection** in sensor data using various algorithms.
2. **Recommendation Generation** using Bayesian-based rules.

The API processes incoming sensor data, detects anomalies, and provides rule-based recommendations.

## Features
- **Anomaly Detection**: Supports multiple algorithms like STL, ARIMA, SARIMA, DBSCAN, etc.
- **Recommendations**: Uses Bayesian analysis to generate recommendations.
- **Dynamic Configuration**: Allows configuration of algorithms through a JSON file.
- **Error Handling**: Handles various errors including invalid input and internal server errors.

## File Structure
```
.
+---algorithms
|   |   __init__.py
|   |
|   +---anomaly_detection
|   |   |   arima_algorithm.py
|   |   |   dbscan_algorithm.py
|   |   |   sarima_algorithm.py
|   |   |   stl_algorithm.py
|   |   |   __init__.py
|   |
|   +---recommendations
|   |   |   bayesian.py
|   |   |   collaborative.py
|   |   |   __init__.py
|
+---app
|   |   models.py
|   |   routes.py
|   |   __init__.py
|   |
|   +---services
|   |       sensor_service.py
|   |       __init__.py
|
+---config
|   |   config.json
|   |   config.py
|   |   __init__.py
|
+---core
|   |   base_anomaly.py
|   |   base_manager.py
|   |   base_recommendation.py
|   |   __init__.py
|
+---data
|   +---csv
|   |       DailyDelhiClimateTrain.csv
|   |       mock_data.csv
|   |       sensor_data.csv
|   |
|   +---logs
|           sensor_data.log
|
+---managers
|   |   anomaly_manager.py
|   |   recommendation_manager.py
|   |   __init__.py
|
+---tests
+---utils
|   |   csv_manager.py
|   |   data_manager.py
|   |   logger.py
|   |   __init__.py
|

```

## Requirements
- Python 3.11
- Flask
- NumPy
- pandas
- statsmodels
- scikit-learn
- pgmpy

## Setup Instructions
### Prerequisites
1. Install Python 3.11.
2. Ensure pip is installed.

### Installation
1. Clone the repository.
2. Navigate to the project directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
- **Dataset**: Ensure the dataset (`DailyDelhiClimateTrain.csv`) is available in the `data/csv` directory.
- **Configuration File**: Update the `config.json` file to specify the algorithms to use for anomaly detection.

### Running the Application
1. Start the Flask server:
   ```bash
   python server.py
   ```
2. The server will start on `http://127.0.0.1:5000`.

## API Endpoints
### 1. Detect Anomalies
**URL**: `/detect_anomalies`  
**Method**: `POST`  
**Content-Type**: `application/json`

#### Request Body
```json
{
  "sensor_values": [
    {
      "timestamp": "2023-11-28T10:00:00",
      "sensor_value": 23.4
    },
    {
      "timestamp": "2023-11-28T11:00:00",
      "sensor_value": 24.1
    }
  ]
}
```

#### Response
- **Success (200)**:
```json
{
  "anomalies": [
    {
      "date": "2023-11-28",
      "meantemp": 23.4,
      "is_anomaly": true
    }
  ]
}
```
- **Error (400)**:
```json
{
  "error": "No valid sensor values found in the request"
}
```

### 2. Recommend Rules
**URL**: `/recommend_rules`  
**Method**: `GET`

#### Response
- **Success (200)**:
```json
[
  "Rule 1: If temperature > 30, then increase AC usage",
  "Rule 2: If humidity < 40%, then activate humidifier"
]
```
- **Error (500)**:
```json
{
  "error": "Internal Server Error: [error message]"
}
```

## Error Handling
- **Invalid Content-Type**: Returns `415` with an appropriate message.
- **Missing or Invalid Data**: Returns `400`.
- **Internal Server Errors**: Returns `500` with details.

## Contributions
Contributions are welcome. Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature/fix.
3. Submit a pull request.

## License
This project is licensed under the MIT License.

---

Developed with ❤️ using Flask.


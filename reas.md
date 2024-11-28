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

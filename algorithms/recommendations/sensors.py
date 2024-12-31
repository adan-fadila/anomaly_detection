# sensors.py
from abc import ABC, abstractmethod
import numpy as np
from algorithms.recommendations.constants import SensorTypes  # Import the required constants

class Sensor(ABC):
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def bins(self):
        pass

    @abstractmethod
    def labels(self):
        pass


class TemperatureSensor(Sensor):
    def name(self):
        return SensorTypes.TEMPERATURE.value  # Use the constant value

    def bins(self):
        return [-np.inf, 15, 20, 25, 32, np.inf]

    def labels(self):
        return [1, 2, 3, 4, 5]


class HumiditySensor(Sensor):
    def name(self):
        return SensorTypes.HUMIDITY.value

    def bins(self):
        return [-np.inf, 30, 60, 90, np.inf]

    def labels(self):
        return [1, 2, 3, 4]


class DistanceSensor(Sensor):
    def name(self):
        return SensorTypes.DISTANCE.value

    def bins(self):
        return [-np.inf, 0.01, 20, np.inf]

    def labels(self):
        return [1, 2, 3]


class SoilSensor(Sensor):
    def name(self):
        return SensorTypes.SOIL.value

    def bins(self):
        return [1850, 2200, 2800]

    def labels(self):
        return [1, 2]


class SeasonSensor(Sensor):
    def name(self):
        return SensorTypes.SEASON.value

    def bins(self):
        return None

    def labels(self):
        return None

    def transform_to_integer(self, season_str):
        season_mapping = {'winter': 1, 'spring': 2, 'summer': 3, 'fall': 4}
        return season_mapping.get(season_str, None)



class SensorFactory:
    @staticmethod
    def create_sensor(sensor_type):
        if sensor_type == SensorTypes.TEMPERATURE.value:
            return TemperatureSensor()
        elif sensor_type == SensorTypes.HUMIDITY.value:
            return HumiditySensor()
        elif sensor_type == SensorTypes.DISTANCE.value:
            return DistanceSensor()
        elif sensor_type == SensorTypes.SOIL.value:
            return SoilSensor()
        elif sensor_type == SensorTypes.SEASON.value:
            return SeasonSensor()
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")
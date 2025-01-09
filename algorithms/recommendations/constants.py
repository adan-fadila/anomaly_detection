# constants.py

from enum import Enum, auto

class SensorTypes(Enum):
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    DISTANCE = "distance"
    SOIL = "soil"
    SEASON = "season"
    HOUR = "hour"

class DeviceTypes(Enum):
    LIGHTS = "lights"
    FAN = "fan"
    AC_STATUS = "ac_status"
    HEATER_SWITCH = "heater_switch"
    LAUNDRY_MACHINE = "laundry_machine"
    PUMP = "pump"

class Seasons(Enum):
    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"

DEVICE_THRESHOLD_URL = "http://localhost:3001/devices_with_thresholds"
MIN_EVIDENCE_STRENGTH_THRESHOLD = 0.001
DEFAULT_DEVICE_THRESHOLD = 0.6
DEFAULT_AVERAGE_DURATION = 1

# New constants
DATA_FILENAME = "mock_data.csv"
MIN_CORRELATION_THRESHOLD = 0.3
DEFAULT_DEVICE = 'ac_energy'
DEFAULT_TIME_RANGE = 'daily'
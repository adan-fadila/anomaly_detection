from algorithms.recommendations.devices import DeviceFactory  # Import the required constants

from algorithms.recommendations.constants import SensorTypes ,DeviceTypes # Import the required constants

class Manager:
    @staticmethod
    def get_list_of_devices():
        device_types = [device.value for device in DeviceTypes]  # Get the list from the Enum
        return [DeviceFactory.create_device(device_type).name() for device_type in device_types]

    @staticmethod
    def get_list_of_devices_with_duration_postfix():
        return [DeviceFactory.create_device(device_type).with_duration_postfix() for device_type in Manager.get_list_of_devices()]

    @staticmethod
    def get_list_of_sensor_values():
        return [sensor.value for sensor in SensorTypes]  # Get the list from the Enum
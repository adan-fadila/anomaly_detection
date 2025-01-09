# devices.py
from abc import ABC, abstractmethod
from algorithms.recommendations.constants import DeviceTypes  # Import the required constants

class Device(ABC):
    @abstractmethod
    def name(self):
        pass

    def with_duration_postfix(self):
        return self.name() + "_duration"


class LightDevice(Device):
    def name(self):
        return DeviceTypes.LIGHTS.value  # Use the constant value


class FanDevice(Device):
    def name(self):
        return DeviceTypes.FAN.value


class ACStatusDevice(Device):
    def name(self):
        return DeviceTypes.AC_STATUS.value


class HeaterSwitchDevice(Device):
    def name(self):
        return DeviceTypes.HEATER_SWITCH.value


class LaundryMachineDevice(Device):
    def name(self):
        return DeviceTypes.LAUNDRY_MACHINE.value


class PumpDevice(Device):
    def name(self):
        return DeviceTypes.PUMP.value


class DeviceFactory:
    @staticmethod
    def create_device(device_type):
        if device_type == DeviceTypes.LIGHTS.value:
            return LightDevice()
        elif device_type == DeviceTypes.FAN.value:
            return FanDevice()
        elif device_type == DeviceTypes.AC_STATUS.value:
            return ACStatusDevice()
        elif device_type == DeviceTypes.HEATER_SWITCH.value:
            return HeaterSwitchDevice()
        elif device_type == DeviceTypes.LAUNDRY_MACHINE.value:
            return LaundryMachineDevice()
        elif device_type == DeviceTypes.PUMP.value:
            return PumpDevice()
        else:
            raise ValueError(f"Unknown device type: {device_type}")
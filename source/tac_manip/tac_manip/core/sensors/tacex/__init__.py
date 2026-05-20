"""
Python module for simulating GelSight sensors inside Isaac Sim/Lab
"""

from .gelsight_sensor import GelSightSensor
from .gelsight_sensor_cfg import GelSightSensorCfg
from .gelsight_sensor_data import GelSightSensorData

# Register UI extensions (optional, only if omni.ui is available).
try:
    from .ui_extension_example import UsdrtExamplePythonExtension
    __all__ = ["GelSightSensor", "GelSightSensorCfg", "GelSightSensorData", "UsdrtExamplePythonExtension"]
except (ImportError, ModuleNotFoundError):
    # UI extensions require Isaac Sim to be fully initialized
    # Skip if omni.ui is not available (e.g., during early imports)
    __all__ = ["GelSightSensor", "GelSightSensorCfg", "GelSightSensorData"]

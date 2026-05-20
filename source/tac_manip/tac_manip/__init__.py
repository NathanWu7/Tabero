"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions (optional, only if omni.ui is available).
try:
    from .ui_extension_example import *
except (ImportError, ModuleNotFoundError):
    # UI extensions require Isaac Sim to be fully initialized
    # Skip if omni.ui is not available (e.g., during early imports)
    pass

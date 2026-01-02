# 1. Define version number
__version__ = "1.0.0"

# 2. Expose submodules
from . import io
from . import generators
from . import consensus
from . import metrics
from . import analysis
from . import pipelines
from . import grid
from . import utils
from . import applications

__all__ = [
    "io",
    "generators",
    "consensus",
    "metrics",
    "analysis",
    "pipelines",
    "grid",
    "utils",
    "applications"
]

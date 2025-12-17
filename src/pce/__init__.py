# 1. 定义版本号
__version__ = "1.0.0"

# 2. 暴露子模块
# 这样用户就可以使用 pce.xxx.xxx
from . import io
from . import generators
from . import consensus
from . import metrics
from . import analysis
from . import pipelines
from . import grid
from . import utils

# 定义当用户使用 'from pce import *' 时导出什么
__all__ = [
    "io",
    "generators",
    "consensus",
    "metrics",
    "analysis",
    "pipelines",
    "grid",
    "utils"
]

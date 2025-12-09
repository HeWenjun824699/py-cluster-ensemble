# 1. 定义版本号
__version__ = "0.1.0"

# 2. 暴露子模块
# 这样用户就可以使用 pce.generators.xxx
from . import generators
from . import consensus
from . import metrics
from . import plots

# 定义当用户使用 'from pce import *' 时导出什么
__all__ = ["generators", "consensus", "metrics", "plots"]

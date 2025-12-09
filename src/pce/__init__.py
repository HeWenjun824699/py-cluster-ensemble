# 1. 定义版本号
__version__ = "0.1.0"

# 2. 暴露子模块
# 这样用户就可以使用 pce.generation.xxx
from . import generation
from . import method
from . import metrics
from . import visualization

# 定义当用户使用 'from pce import *' 时导出什么
__all__ = ["generation", "method", "metrics", "visualization"]

# py-cluster-ensemble (PCE)

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue)](https://pypi.org/project/py-cluster-ensemble/) [![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **A Comprehensive Python Toolkit for Cluster Ensemble Generation, Consensus, and Automated Experimentation.**

**py-cluster-ensemble (PCE)** 是一款专为科研与学术界打造的 Python 聚类集成（Cluster Ensemble）通用框架。

它致力于解决 Python 生态中缺乏统一聚类集成工具的痛点，提供从 **基聚类生成 (Generation)**、**集成共识 (Consensus)** 到 **结果评估 (Evaluation)** 的标准化流水线。PCE 特别针对科研实验场景进行了深度优化，完美兼容 MATLAB (`.mat`) 数据交互，并内置 **智能网格搜索**、**自动化批处理** 及 **论文级可视化** 模块，旨在打造从 MATLAB 到 Python 的无缝迁移体验。

---

## 📋 目录 (Table of Contents) 

- [安装 (Installation)](#install) 
- [快速开始 (Quick Start)](#quickstart)
- [核心模块 API (API Reference)](#api_reference) 
  - [1. 输入输出 (IO)](#io)    
  - [2. 基聚类生成器 (Generators)](#generators) 
  - [3. 集成算法 (Consensus)](#consensus) 
  - [4. 评估指标 (Metrics)](#metrics)
  - [5. 分析模块 (Analysis)](#analysis)
  - [6. 流水线 (Pipelines)](#pipelines)
  - [7. 网格搜索 (Grid)](#grid)
  - [8. 工具模块 (Utils)](#utils)
  
- [项目规划 (Roadmap)](#roadmap) 

---

## <span id="install">📦 安装 (Installation)</span>

使用 pip 安装 (推荐)
~~~
pip install py-cluster-ensemble
~~~

---

## <span id="quickstart">🚀 快速开始 (Quick Start)</span>

### 场景 A: 一键批处理 (推荐)

使用 consensus_batch 自动扫描目录下的所有 .mat 数据集，生成基聚类、运行集成算法并导出csv、xlsx、mat格式的数据。

~~~
from pce.pipelines import consensus_batch

# 运行流水线
consensus_batch(
    input_dir='./data',            # 数据集目录
    output_dir='./results',        # 结果保存目录
    consensus_method='cspa',       # 集成算法: 'cspa', 'mcla', 'hgpa'
    generator_method='cdkmeans',   # 生成器: 'cdkmeans', 'litekmeans'
    nBase=20,                      # 每次集成的基聚类数
    nRepeat=10,                    # 实验重复轮数
    save_format='csv',             # 保存格式: 'xlsx', 'csv', 'mat'
    overwrite=True                 # 是否覆盖已有结果
)

# consensus_batch 仅 input_dir 为必须参数，其他参数的默认值可在参数说明中查看
~~~

### 场景 B: 模块化分步调用

如果您需要更细粒度的控制，可以独立调用各个模块：

~~~
import pce.io as io
import pce.generators as gen
import pce.consensus as con
import pce.metrics as met

# 1. 加载数据 (自动处理 .mat 格式)
X, Y = io.load_mat_X_Y('data/isolet.mat')

# 2. 生成基聚类 (使用 CDK-Means)
BPs, _ = gen.cdkmeans(X, Y, nBase=200)

# 3. 执行集成 (使用 CSPA)
# 将 200 个基聚类切分为 10 组，每组 20 个进行实验
labels_list, _ = con.cspa(BPs, Y, nBase=20, nRepeat=10)

# 4. 评估结果
results = met.evaluation_batch(labels_list, Y)

# 5. 保存结果为 Excel (保留 4 位小数格式)
io.save_xlsx(results, 'output/isolet_report.xlsx')
~~~

### 场景 C: 超参数网格搜索(Grid Search)

针对科研实验设计的网格搜索模块，支持对集成算法和基聚类生成器进行参数扫描。<br>
`GridSearcher` 会自动计算参数的笛卡尔积组合，并智能跳过无效的参数配置。

~~~
import pce

# 1. 准备路径
input_dir = './data'
output_dir = './grid_results'

# 2. 定义网格参数 (param_grid)
# 字典中的 value 必须是列表，程序会自动生成所有组合
param_grid = {
    'consensus_method': ['cspa', 'mcla'],   # 同时对比多种集成算法
    't': [20, 50, 100],                     # 探究超参数 t 的影响
    'k': [5, 10, 15]                        # 探究超参数 k 的影响
}

# 3. 定义固定参数 (fixed_params)
# 所有实验共用的静态参数
fixed_params = {
    'generator_method': 'cdkmeans',
    'nBase': 20,        # generators, consensus
    'seed': 2024,       # generators, consensus
    'maxiter': 100,     # generators
    'replicates': 1,    # generators
    'nRepeat': 10       # consensus
}

# 4. 初始化并运行
# 结果将生成汇总 CSV 表格及详细的 JSON/Log 文件
searcher = pce.grid.GridSearcher(input_dir, output_dir)
searcher.run(param_grid, fixed_params)

# 小贴士: 不确定某个算法支持哪些参数？
# 使用工具函数查看参数列表:
# pce.utils.show_function_params('cspa', module_type='consensus')
~~~

### 场景 D: 论文级可视化 (Visualization)

PCE 内置了符合学术标准的绘图模块，能够直接利用上述场景生成的数据，一键绘制高质量插图。**所有绘图结果均支持通过 `save_path` 后缀自动保存为 `.png` (位图) 或 `.pdf` (矢量图)**。

~~~
import pce.io as io
import pce.analysis as ana

# 1. 降维散点图 (t-SNE/PCA)
# 适用场景: 原始数据分布展示 / 聚类结果可视化
X, Y = io.load_mat_X_Y('data/isolet.mat')
ana.plot_2d_scatter(
    X, Y, 
    method='tsne', 
    title='Ground Truth Visualization (t-SNE)',
    save_path='output/tsne_plot.png'
)

# 2. 共协矩阵热力图 (Co-association Heatmap)
# 适用场景: 观察集成的一致性结构 (需先运行场景 B 生成 BPs)
BPs, Y = io.load_mat_BPs_Y('data/base_partitions.mat')
ana.plot_coassociation_heatmap(
    BPs, Y,
    title='Ensemble Consensus Matrix',
    save_path='output/heatmap.png'
)

# 3. 性能趋势折线图 (Metric Trace)
# 适用场景: 展示场景 B 中 nRepeat 次实验的稳定性
# results_list 是场景 B 中 evaluation_batch 的返回值
ana.plot_metric_line(
    results_list, 
    metrics=['ACC', 'NMI', 'ARI'], 
    xlabel='Run ID',
    title='Performance Stability over 10 Runs',
    save_path='output/trace_plot.png'
)

# 4. 参数敏感度分析 (Sensitivity Analysis)
# 适用场景: 分析场景 C 网格搜索中某个参数(如 t)对性能的影响
# csv_file 是场景 C 生成的汇总表
ana.plot_parameter_sensitivity(
    csv_file='grid_results/isolet_summary.csv',
    target_param='t',    # X轴: 变化的参数
    metric='ACC',        # Y轴: 观察的指标
    fixed_params={'k': 10}, # 控制变量: 固定其他参数
    save_path='output/sensitivity_t.png'
)
~~~

---

## <span id="api_reference">📚 核心模块 API (API Reference)</span>

## <span id="io">📂 1. 输入输出 (pce.io)</span>

**数据接口模块，** 负责处理与 `.mat` 文件的交互及结果持久化，自动解决 MATLAB v7.3 格式兼容性问题，并支持将实验结果导出为 `CSV`、`Excel` 或 `MAT` 格式。

<details>
<summary><strong>🔽 点击查看详细参数列表 (Click to expand)</strong></summary>

### 1.1 load_mat_X_Y 参数和返回值说明

用于读取包含原始特征矩阵 X 和标签 Y 的 MATLAB 数据文件，支持自动识别变量名及数据类型转换。

#### 1.1.1 参数 (Parameters)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`file_path`** | **`Union[str, Path]`** | **必填** | **输入 .mat 文件的路径**<br>支持字符串或 `pathlib.Path` 对象，代码会自动识别并处理标准 MATLAB 格式以及 v7.3 (HDF5) 格式 |
| `ensure_x_float` | `bool` | `True` | **强制转换特征矩阵为浮点数**<br>如果为 `True`，会将读取到的特征矩阵 `X` 转换为 `np.float64` 类型，这对大多数聚类算法（如 K-Means）的数值计算稳定性至关重要 |
| `flatten_y` | `bool` | `True` | **展平标签向量**<br>如果为 `True`，会将读取到的标签 `Y` 从二维列向量 `(n_samples, 1)` 展平为一维数组 `(n_samples,)`，符合 Scikit-learn 等 Python 机器学习库的标准输入格式 |

#### 1.1.2 返回值 (Returns)

| 变量名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **`X`** | `np.ndarray` | **特征矩阵**<br>形状为 `(n_samples, n_features)`<br>代码会自动尝试匹配常见的变量名（如 `'X'`, `'data'`, `'fea'`, `'features'` 等） |
| **`Y`** | `np.ndarray` | **标签向量**<br>形状通常为 `(n_samples,)`<br>代码会自动尝试匹配常见的变量名（如 `'Y'`, `'label'`, `'gnd'` 等），且会自动将浮点型整数标签安全转换为 `np.int64` 类型 |

### 1.2 load_mat_BPs_Y 参数和返回值说明

用于读取包含基聚类矩阵 BPs 和标签 Y 的 MATLAB 数据文件，常用于加载外部生成的基聚类结果。

#### 1.2.1 参数 (Parameters)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`file_path`** | **`Union[str, Path]`** | **必填** | **输入 .mat 文件的路径**<br>同上，支持自动版本识别 |
| `fix_matlab_index` | `bool` | `True` | **修正 MATLAB 索引**<br>如果为 `True` 且检测到 `BPs` 中的最小值为 1，函数会自动将所有值减 1，从而将 MATLAB 的 1-based 索引转换为 Python 的 0-based 索引 |
| `flatten_y` | `bool` | `True` | **展平标签向量**<br>同上，将标签转为一维数组 |

#### 1.2.2 返回值 (Returns)

| 变量名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **`BPs`** | `np.ndarray` | **基聚类矩阵**<br>形状为 `(n_samples, n_estimators)`<br>数据会被强制转换为 `np.int64`，如果触发了 `fix_matlab_index`，返回的数据范围将从 0 开始 |
| **`Y`** | `np.ndarray` | **标签向量**<br>形状通常为 `(n_samples,)`<br>同 `load_mat_X_Y` 中的 Y 处理逻辑 |

### 1.3 save_base_mat 参数说明

将生成的基聚类矩阵和真实标签保存为 MATLAB 兼容的 `.mat` 文件，便于跨平台验证。

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`BPs`** | **`np.ndarray`** | **必填** | **基聚类矩阵 (Base Partitions)**<br>通常是 $N \times M$ 的矩阵（N=样本数，M=基聚类器数量）。<br>函数内部会自动将其转换为 **`float64` (double)** 类型以兼容 MATLAB，在 `.mat` 文件中变量名为 `'BPs'`。 |
| **`Y`** | **`np.ndarray`** | **必填** | **真实标签 (True Labels)**<br>样本的真实类别标签。<br>函数内部会自动执行以下操作：<br>1. 转换为 **`float64` (double)** 类型<br>2. 强制重塑为 **$N \times 1$ 的列向量**<br>在 `.mat` 文件中变量名为 `'Y'`。 |
| **`output_path`** | **`str`** | **必填** | **输出路径**<br>支持智能识别：<br>**1. 目录路径** (以 `/` 或 `\` 结尾，或已存在的文件夹)：文件将保存到该目录下，文件名由 `default_name` 指定<br>**2. 文件路径** (如 `output/data.mat`)：直接保存为该文件，代码会自动补全 `.mat` 后缀并创建不存在的父目录 |
| `default_name` | `str` | `"base.mat"` | **默认文件名**<br>仅当 `output_path` 被判定为目录时使用 |

### 1.4 save_results_csv 参数说明

将实验结果列表导出为 CSV 文本文件，适合快速查看或作为轻量级数据交换格式。

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`data`** | **`List[Dict]`** | **必填** | **评估结果数据**<br>通常是包含多轮实验结果的字典列表。列表中的每个字典代表一行数据，字典的 Key 将成为 CSV 的列名 |
| **`output_path`** | **`str`** | **必填** | **输出路径**<br>支持智能识别：<br>**1. 目录路径** (以 `/` 或 `\` 结尾，或已存在的文件夹)：文件将保存到该目录下，文件名由 `default_name` 指定<br>**2. 文件路径** (如 `output/res.csv`)：直接保存为该文件，代码会自动创建不存在的父目录 |
| `default_name` | `str` | `"result.csv"` | **默认文件名**<br>仅当 `output_path` 被判定为目录时使用 |
| `add_summary` | `bool` | `True` | **是否追加统计摘要**<br>如果为 `True`，会在原始数据后插入一个**空行**，然后计算并追加所有数值列的 **均值 (Mean)** 和 **标准差 (Std)** |
| `float_format` | `str` | `"%.4f"` | **浮点数格式控制**<br>指定写入 CSV 时浮点数的精度。默认保留 4 位小数 |

### 1.5 save_results_xlsx 参数说明

将实验结果导出为 Excel 文件，支持数值类型保留，便于后续在 Excel 中进行公式计算或制图。

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`data`** | **`List[Dict]`** | **必填** | **评估结果数据**<br>通常是包含多轮实验结果的字典列表。列表中的每个字典代表一行数据，字典的 Key 将成为 Excel 的列名 |
| **`output_path`** | **`str`** | **必填** | **输出路径**<br>支持智能识别：<br>**1. 目录路径** (以 `/` 或 `\` 结尾，或已存在的文件夹)：文件将保存到该目录下，文件名由 `default_name` 指定<br>**2. 文件路径** (如 `output/res.xlsx`)：直接保存为该文件。代码会自动创建不存在的父目录，并强制后缀为 `.xlsx` |
| `default_name` | `str` | `"result.xlsx"` | **默认文件名**<br>仅当 `output_path` 被判定为目录时使用 |
| `add_summary` | `bool` | `True` | **是否追加统计摘要**<br>如果为 `True`，会在原始数据后插入一个**空行**，然后计算并追加所有数值列的 **均值 (Mean)** 和 **标准差 (Std)** |
| `excel_format` | `str` | `"0.0000"` | **Excel 数值格式控制**<br>指定 Excel 单元格的数字显示格式字符串。例如 `"0.0000"` 表示显示 4 位小数<br>相比 CSV，此方式能保持单元格为**数值类型**（便于在 Excel 中进行求和等后续计算），而非纯文本 |

### 1.6 save_results_mat 参数说明

将实验结果保存为 MATLAB 结构化数据，方便在 MATLAB 环境中加载并进行后续分析。

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`data`** | **`List[Dict]`** | **必填** | **评估结果数据**<br>通常是包含多轮实验结果的字典列表。函数会将字典中的数值提取并转换为 NumPy 矩阵保存到 `.mat` 文件变量 `result` 中 |
| **`output_path**`** | **`str`** | **必填** | **输出路径**<br>支持智能识别：<br>**1. 目录路径** (以 `/` 或 `\` 结尾，或已存在的文件夹)：文件将保存到该目录下，文件名由 `default_name` 指定<br>**2. 文件路径** (如 `output/res.mat`)：直接保存为该文件。代码会自动创建不存在的父目录，并强制后缀为 `.mat` |
| `default_name` | `str` | `"result.mat"` | **默认文件名**<br>仅当 `output_path` 被判定为目录时使用 |
| `add_summary` | `bool` | `True` | **是否追加统计摘要**<br>如果为 `True`，会计算所有结果的 **均值 (Mean)** 和 **标准差 (Std)**，并分别以变量名 `result_summary` 和 `result_summary_std` 保存到 `.mat` 文件中 |

</details>

## <span id="generators">⚙️ 2. 基聚类生成器 (pce.generators)</span>

**基聚类生成模块，** 封装了多种基聚类生成逻辑，支持通过控制随机初始化、迭代策略及子空间扰动，从原始特征矩阵构建具有多样性的基聚类池。

<details>
<summary><strong>🔽 点击查看详细参数列表 (Click to expand)</strong></summary>

### 2.1 litekmeans 参数和返回值说明

基于 LiteKMeans 的高速基聚类生成器，适用于大规模数据的快速聚类池构建。

#### 2.1.1 参数 (Parameters)

| 参数名       | 类型             | 默认值   | 说明                                                         |
| :----------- | :--------------- | :------- | :----------------------------------------------------------- |
| **`X`**      | **`np.ndarray`** | **必填** | **输入特征矩阵**<br>形状通常为 `(n_samples, n_features)`, 算法将在此数据上运行 KMeans |
| **`Y`**      | **`np.ndarray`** | **必填** | **真实标签向量**<br>形状通常为 `(n_samples,)` 或 `(n_samples, 1)` <br>**用途：**代码内部使用 len(np.unique(Y)) 来确定数据集的真实类别数 nCluster，进而计算随机 K 值的范围 [minCluster, maxCluster] |
| `nBase`      | `int`            | `200`    | 生成基聚类（Base Partitions）的数量，即结果矩阵的列数        |
| `seed`       | `int`            | `2024`   | 随机种子，用于控制 K 值的随机选择和算法初始化，保证结果可复现 |
| `maxiter`    | `int`            | `100`    | 聚类算法（LiteKMeans）的最大迭代次数                         |
| `replicates` | `int`            | `1`      | 每次聚类尝试运行的重复次数，算法会返回其中目标函数最优的一次结果 |

####  2.1.2 返回值 (Returns)

| 变量名 | 类型          | 说明                                                                                                                     |
|:------|:-------------|:-------------------------------------------------------------------------------------------------------------------------|
| `BPs` | `np.ndarray` | **基聚类矩阵 (Base Partitions)**<br>形状为 (n_samples, nBase)。每一列代表一次 KMeans 聚类的结果标签（注意：代码中已对标签进行了 +1 处理，适应 MATLAB 风格索引，或便于区分）   |
| `Y`   | `np.ndarray` | **真实标签**<br>原样返回传入的 Y，便于后续流程直接调用                                                                                             |

### 2.2 cdkmeans 参数和返回值说明

基于约束条件（Constraints）的差异化 K-Means 生成器，旨在通过增加基聚类间的差异性来提升集成效果。

#### 2.2.1 参数 (Parameters)

| 参数名       | 类型             | 默认值   | 说明                                                         |
| :----------- | :--------------- | :------- | :----------------------------------------------------------- |
| **`X`**      | **`np.ndarray`** | **必填** | **输入特征矩阵**<br>形状通常为` (n_samples, n_features)`, 算法将在此数据上运行 KMeans 初始化及 CDKM 优化 |
| **`Y`**      | **`np.ndarray`** | **必填** | **真实标签向量**<br>形状通常为 `(n_samples,)` 或 `(n_samples, 1)` <br>**用途：**代码内部使用 len(np.unique(Y)) 来确定数据集的真实类别数 nCluster，进而计算随机 K 值的范围 [minCluster, maxCluster] |
| `nBase`      | `int`            | `200`    | 生成基聚类（Base Partitions）的数量，即结果矩阵的列数        |
| `seed`       | `int`            | `2024`   | 随机种子，用于控制 K 值的随机选择和算法初始化，保证结果可复现 |
| `maxiter`    | `int`            | `100`    | 聚类算法（LiteKMeans）的最大迭代次数                         |
| `replicates` | `int`            | `1`      | 每次聚类尝试运行的重复次数，算法会返回其中目标函数最优的一次结果 |

#### 2.2.2 返回值 (Returns)

| 变量名 | 类型          | 说明                                                                                                                     |
|:------|:-------------|:-------------------------------------------------------------------------------------------------------------------------|
| `BPs` | `np.ndarray` | **基聚类矩阵 (Base Partitions)**<br>形状为 (n_samples, nBase)。每一列代表一次 KMeans 聚类的结果标签（注意：代码中已对标签进行了 +1 处理，适应 MATLAB 风格索引，或便于区分）   |
| `Y`   | `np.ndarray` | **真实标签**<br>原样返回传入的 Y，便于后续流程直接调用                                                                                             |

### 2.3 litekmeans_old 参数说明

LiteKMeans 的文件接口封装，直接处理 `.mat` 文件输入输出，适用于不依赖内存变量的独立脚本调用。

| 参数名           | 类型      | 默认值         | 说明                                                         |
| :-------------- | :-------- |:------------| :----------------------------------------------------------- |
| **`file_path`** | **`str`** | **必填**     | **输入数据文件的完整路径（支持 `.mat` 格式，兼容 v7.3）**    |
| `output_path`   | `str`     | `None`      | 输出 `.mat` 文件的保存路径（包含文件名）<br>如果为 `None`，默认保存到输入文件同目录下，文件名为 `[原文件名]_LKM[nBase].mat` |
| `nBase`         | `int`     | `200`       | 生成基聚类（Base Partitions）的数量，即结果矩阵的列数        |
| `seed`          | `int`     | `2024`      | 随机种子，用于控制 K 值的随机选择和算法初始化，保证结果可复现 |
| `maxiter`       | `int`     | `100`       | 聚类算法（LiteKMeans）的最大迭代次数                         |
| `replicates`    | `int`     | `1`         | 每次聚类尝试运行的重复次数，算法会返回其中目标函数最优的一次结果 |

### 2.4 cdkmeans_old 参数说明

CDK-Means 的文件接口封装，直接处理 `.mat` 文件输入输出。

| 参数名           | 类型      | 默认值         | 说明                                                         |
| :--------------| :-------- |:------------| :----------------------------------------------------------- |
| **`file_path`** | **`str`** | **必填**     | **输入数据文件的完整路径（支持 `.mat` 格式，兼容 v7.3）**    |
| `output_path`   | `str`     | `None`      | 输出 `.mat` 文件的保存路径（包含文件名）<br>如果为 `None`，默认保存到输入文件同目录下，文件名为 `[原文件名]_CDKM[nBase].mat` |
| `nBase`         | `int`     | `200`       | 生成基聚类（Base Partitions）的数量，即结果矩阵的列数        |
| `seed`          | `int`     | `2024`      | 随机种子，用于控制 K 值的随机选择和算法初始化，保证结果可复现 |
| `maxiter`       | `int`     | `100`       | 聚类算法（LiteKMeans）的最大迭代次数                         |
| `replicates`    | `int`     | `1`         | 每次聚类尝试运行的重复次数，算法会返回其中目标函数最优的一次结果 |

</details>

## <span id="consensus">🤝 3. 集成算法 (pce.consensus)</span>

**核心集成算法模块，** 实现了从基聚类矩阵推导最终共识划分的逻辑，涵盖基于共协矩阵（CSPA）、基于元聚类（MCLA）及基于超图分割（HGPA）等经典集成算法。

<details>
<summary><strong>🔽 点击查看详细参数列表 (Click to expand)</strong></summary>

### 3.1 cspa 参数和返回值说明

基于聚类集成共协矩阵（Cluster-based Similarity Partitioning Algorithm）的集成方法，利用共协矩阵度量样本对的成对相似性，并通过谱聚类获得最终划分。

#### 3.1.1 参数 (Parameters)

| 参数名     | 类型      | 默认值         | 说明                                                                                                                                                       |
|:----------| :-------- |:------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`BPs`** | **`np.ndarray`** | **必填** | **基聚类矩阵 (Base Partitions)**<br>形状通常为 `(n_samples, n_total_clusterings)`<br>每一列代表一个基聚类器的结果, 代码内部会自动检测并处理 MATLAB 风格的 1-based 索引（将其转换为 Python 的 0-based 索引） |
| **`Y`**   | **`np.ndarray`** | **必填** | **真实标签向量**<br>形状为 `(n_samples,)` 或 `(n_samples, 1)` <br>**用途：**代码内部使用 `len(np.unique(Y))` 来确定最终集成聚类的目标类别数 `K(nCluster)`     |
| `nBase`   | `int`     | `20`        | **单次实验**使用的基聚类数量（切片大小）<br>例如：池中共有 200 个基聚类，设为 20 表示每次实验只使用其中 20 个来进行集成                                                                                   |
| `nRepeat` | `int`     | `10`        | 实验重复次数，程序会进行 `nRepeat` 次独立实验，所需的基聚类总列数 = `nBase` × `nRepeat`                                                                                            |
| `seed`    | `int`     | `2024`      | 随机种子，用于控制 CSPA 内部谱聚类（Spectral Clustering）的初始化状态，保证可复现性                                                                                                   |

#### 3.1.2 返回值 (Returns)

| 变量名        | 类型               | 说明                                                         |
| :------------ | :----------------- | :----------------------------------------------------------- |
| `labels_list` | `List[np.ndarray]` | **预测标签列表**<br>包含 `nRepeat` 个元素的列表，每个元素是一个形状为 `(n_samples,)` 的一维 NumPy 数组，代表某次实验的 CSPA 集成结果 |
| `Y`           | `np.ndarray`       | **真实标签**<br>原样返回传入的 `Y`，便于后续流程直接调用     |

### 3.2 mcla 参数说明和返回值说明

基于元聚类算法（Meta-Clustering Algorithm）的集成方法，通过对基聚类的簇标记（Label）进行聚类来解决标记对应问题，适合处理大规模数据。

#### 3.2.1 参数 (Parameters)

| 参数名      | 类型      | 默认值         | 说明                                                        |
|:-----------| :-------- |:------------| :----------------------------------------------------------- |
| **`BPs`**  | **`np.ndarray`** | **必填** | **基聚类矩阵 (Base Partitions)**<br>形状通常为 `(n_samples, n_total_clusterings)`<br>每一列代表一个基聚类器的结果, 代码内部会自动检测并处理 MATLAB 风格的 1-based 索引（将其转换为 Python 的 0-based 索引） |
| **`Y`**    | **`np.ndarray`** | **必填** | **真实标签向量**<br>形状为 `(n_samples,)` 或 `(n_samples, 1)` <br>**用途：**代码内部使用 `len(np.unique(Y))` 来确定最终集成聚类的目标类别数 `K(nCluster)` |
| `nBase`    | `int`     | `20`        | **单次实验**使用的基聚类数量（切片大小）<br>例如：池中共有 200 个基聚类，设为 20 表示每次实验只使用其中 20 个来进行集成    |
| `nRepeat`  | `int`     | `10`        | 实验重复次数，程序会进行 `nRepeat` 次独立实验，所需的基聚类总列数 = `nBase` × `nRepeat`             |
| `seed`     | `int`     | `2024`      | 随机种子，用于控制 MCLA 内部元聚类（Meta-Clustering）阶段的初始化状态（如谱聚类初始化），保证可复现性             |

#### 3.2.2 返回值 (Returns)

| 变量名        | 类型               | 说明                                                         |
| :------------ | :----------------- | :----------------------------------------------------------- |
| `labels_list` | `List[np.ndarray]` | **预测标签列表**<br>包含 `nRepeat` 个元素的列表，每个元素是一个形状为 `(n_samples,)` 的一维 NumPy 数组，代表某次实验的 MCLA 集成结果 |
| `Y`           | `np.ndarray`       | **真实标签**<br>原样返回传入的 `Y`，便于后续流程直接调用     |

### 3.3 hgpa 参数和返回值说明

基于超图分割（HyperGraph Partitioning Algorithm）的集成方法，将基聚类中的簇建模为超边，通过分割超图来寻找最佳共识划分。

#### 3.3.1 参数 (Parameters)

| 参数名      | 类型      | 默认值         | 说明                                                        |
|:-----------| :-------- |:------------| :----------------------------------------------------------- |
| **`BPs`**  | **`np.ndarray`** | **必填** | **基聚类矩阵 (Base Partitions)**<br>形状通常为 `(n_samples, n_total_clusterings)`<br>每一列代表一个基聚类器的结果, 代码内部会自动检测并处理 MATLAB 风格的 1-based 索引（将其转换为 Python 的 0-based 索引） |
| **`Y`**    | **`np.ndarray`** | **必填** | **真实标签向量**<br>形状为 `(n_samples,)` 或 `(n_samples, 1)` <br>**用途：**代码内部使用 `len(np.unique(Y))` 来确定最终集成聚类的目标类别数 `K(nCluster)` |
| `nBase`    | `int`     | `20`        | **单次实验**使用的基聚类数量（切片大小）<br>例如：池中共有 200 个基聚类，设为 20 表示每次实验只使用其中 20 个来进行集成 |
| `nRepeat`  | `int`     | `10`        | 实验重复次数，程序会进行 `nRepeat` 次独立实验，所需的基聚类总列数 = `nBase` × `nRepeat` |
| `seed`     | `int`     | `2024`      | 随机种子，用于控制 HGPA 内部超图分割（Hypergraph Partitioning）阶段的初始化状态，保证可复现性 |

#### 3.3.2 返回值 (Returns)

| 变量名        | 类型               | 说明                                                         |
| :------------ | :----------------- | :----------------------------------------------------------- |
| `labels_list` | `List[np.ndarray]` | **预测标签列表**<br>包含 `nRepeat` 个元素的列表，每个元素是一个形状为 `(n_samples,)` 的一维 NumPy 数组，代表某次实验的 HGPA 集成结果 |
| `Y`           | `np.ndarray`       | **真实标签**<br>原样返回传入的 `Y`，便于后续流程直接调用     |

### 3.4 cspa_old 参数说明

CSPA 的文件接口封装，直接读取 `.mat` 文件进行计算，适合不依赖 Python 变量环境的独立调用。

| 参数名           | 类型      | 默认值         | 说明                                                        |
| :-------------- | :-------- |:------------| :----------------------------------------------------------- |
| **`file_path`** | **`str`** | **必填**     | **输入数据文件的完整路径（`.mat` 格式）**<br>**文件内必须包含变量 `BPs` (基聚类矩阵) 和 `Y` (真实标签)** |
| `output_path`   | `str`     | `None`      | 输出 `.mat` 文件的保存路径 <br>如果为 `None`，默认在输入文件同级目录下新建 `CSPA_Results/[文件名]/` 保存结果 |
| `nBase`         | `int`     | `20`        | **单次实验**使用的基聚类数量（切片大小）<br>例如：池中共有 200 个基聚类，设为 20 表示每次实验只使用其中 20 个来进行集成 |
| `nRepeat`       | `int`     | `10`        | 实验重复次数，程序会进行 `nRepeat` 次独立实验，所需的基聚类总列数 = `nBase` × `nRepeat` |
| `seed`          | `int`     | `2024`      | 随机种子，用于控制 CSPA 内部谱聚类（Spectral Clustering）的初始化状态，保证可复现性 |

### 3.5 mcla_old 参数说明

MCLA 的文件接口封装，直接读取 `.mat` 文件进行计算。

| 参数名           | 类型      | 默认值         | 说明                                                        |
| :-------------- | :-------- |:------------| :----------------------------------------------------------- |
| **`file_path`** | **`str`** | **必填**     | **输入数据文件的完整路径（`.mat` 格式）**<br>**文件内必须包含变量 `BPs` (基聚类矩阵) 和 `Y` (真实标签)**  |
| `output_path`   | `str`     | `None`      | 输出 `.mat` 文件的保存路径<br>如果为 `None`，默认在输入文件同级目录下新建 `MCLA_Results/[文件名]/` 保存结果 |
| `nBase`         | `int`     | `20`        | **单次实验**使用的基聚类数量（切片大小）<br>例如：池中共有 200 个基聚类，设为 20 表示每次实验只使用其中 20 个来进行集成    |
| `nRepeat`       | `int`     | `10`        | 实验重复次数，程序会进行 `nRepeat` 次独立实验，所需的基聚类总列数 = `nBase` × `nRepeat`             |
| `seed`          | `int`     | `2024`      | 随机种子，用于控制 MCLA 内部元聚类（Meta-Clustering）阶段的初始化状态（如谱聚类初始化），保证可复现性             |

### 3.6 hgpa_old 参数说明

HGPA 的文件接口封装，直接读取 `.mat` 文件进行计算。

| 参数名           | 类型      | 默认值         | 说明                                                        |
| :-------------- | :-------- |:------------| :----------------------------------------------------------- |
| **`file_path`** | **`str`** | **必填**     | **输入数据文件的完整路径（`.mat` 格式）**<br>**文件内必须包含变量 `BPs` (基聚类矩阵) 和 `Y` (真实标签)** |
| `output_path`   | `str`     | `None`      | 输出 `.mat` 文件的保存路径<br>如果为 `None`，默认在输入文件同级目录下新建 `HGPA_Results/[文件名]/` 保存结果 |
| `nBase`         | `int`     | `20`        | **单次实验**使用的基聚类数量（切片大小）<br>例如：池中共有 200 个基聚类，设为 20 表示每次实验只使用其中 20 个来进行集成 |
| `nRepeat`       | `int`     | `10`        | 实验重复次数，程序会进行 `nRepeat` 次独立实验，所需的基聚类总列数 = `nBase` × `nRepeat` |
| `seed`          | `int`     | `2024`      | 随机种子，用于控制 HGPA 内部超图分割（Hypergraph Partitioning）阶段的初始化状态，保证可复现性 |

</details>

## <span id="metrics">📊 4. 评估指标 (pce.metrics)</span>

**指标评估模块，** 提供全套聚类验证指标（包含 NMI, ARI, ACC 等 14 种），支持单次结果验证及批量实验的统计分析，用于量化对比集成算法的有效性。

<details>
<summary><strong>🔽 点击查看详细参数列表 (Click to expand)</strong></summary>

### 4.1 evaluation_single 参数和返回值说明

用于评估单个预测标签向量（Predict Label）与真实标签（Ground Truth）之间的聚类性能，一次性返回 14 种指标。

#### 4.1.1 参数 (Parameters)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | ---- |
| **`y`** | **`np.ndarray`** | **必填** | **预测标签向量**<br>聚类算法输出的预测结果，代码内部会自动展平为一维数组 |
| **`Y`** | **`np.ndarray`** | **必填** | **真实标签向量**<br>数据集的真实标签 (Ground Truth)，用于评估聚类性能，代码内部会自动展平为一维数组 |

#### 4.1.2 返回值 (Returns)

| 变量名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **`res`** | `List[float]` | **评估指标列表**<br>包含14个浮点数的列表，顺序依次为：<br>`[ACC, NMI, Purity, AR, RI, MI, HI, F-Score, Precision, Recall, Entropy, SDCS, RME, Bal]` |

### 4.2 evaluation_batch 参数和返回值说明

用于批量评估多轮实验（例如 nRepeat 次重复实验）的结果，返回包含所有实验指标的列表，适合后续进行均值和方差统计。

#### 4.2.1 参数 (Parameters)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | ---- |
| **`labels`** | **`List[np.ndarray]`** | **必填** | **预测标签列表**<br>包含多个预测结果的列表，列表中的每个元素都是一个形状为 `(n_samples,)` 的一维数组（例如多次实验的聚类结果），函数将遍历此列表逐一评估 |
| **`Y`** | **`np.ndarray`** | **必填** | **真实标签向量**<br>数据集的真实标签 (Ground Truth)，用于评估聚类性能，代码内部会自动展平为一维数组 |

#### 4.2.2 返回值 (Returns)

| 变量名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **`res_list`** | `List[Dict]` | **评估结果列表**<br>列表中的每个元素都是一个字典，对应 `labels` 中每一次预测的评估结果。字典包含以下 14 个 Key：<br>`['ACC', 'NMI', 'Purity', 'AR', 'RI', 'MI', 'HI', 'F-Score', 'Precision', 'Recall', 'Entropy', 'SDCS', 'RME', 'Bal']` |

</details>

## <span id="analysis">📈 5. 分析模块 (pce.analysis)</span>

**可视化分析模块，** 提供符合学术发表标准（Paper-Style）的绘图工具，支持数据降维可视化、共协矩阵热力图、实验性能折线图及参数敏感性分析。

<details>
<summary><strong>🔽 点击查看详细参数列表 (Click to expand)</strong></summary>

### 5.1 plot_2d_scatter 参数说明

用于绘制原始数据或聚类结果的 2D 散点图，若数据维度大于 2，自动调用 t-SNE 或 PCA 进行降维。

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`X`** | **`np.ndarray`** | **必填** | **特征矩阵**<br>形状通常为 `(n_samples, n_features)`，若列数大于 2 将自动进行降维处理 |
| **`labels`** | **`np.ndarray`** | **必填** | **标签向量**<br>形状为 `(n_samples,)`，用于控制散点颜色，支持真实标签或预测标签 |
| `method` | `str` | `'tsne'` | **降维方法**<br>支持 `'tsne'` (默认) 或 `'pca'` |
| `title` | `str` | `None` | **图表标题**<br>若为 `None`，自动生成包含方法名和样本数的默认标题 |
| `save_path` | `str` | `None` | **保存路径**<br>支持 `.png`, `.pdf` 等格式，若路径不存在会自动创建目录 |
| `show` | `bool` | `True` | **显示窗口**<br>是否在绘图结束后调用 `plt.show()` 弹出窗口 |
| `**kwargs` | `Any` | - | **透传参数**<br>传递给 `TSNE` 或 `PCA` 初始化函数的其他关键字参数 |

### 5.2 plot_coassociation_heatmap 参数说明

绘制排序后的共协矩阵（Co-association Matrix）热力图，用于直观评估基聚类的集成一致性。

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`BPs`** | **`np.ndarray`** | **必填** | **基聚类矩阵**<br>形状为 `(n_samples, n_estimators)`，用于计算样本间的 Hamming 距离 |
| **`Y`** | **`np.ndarray`** | **必填** | **真实标签/参考标签**<br>用于对矩阵行列进行排序，使得同类样本在对角线上形成聚集块 |
| `xlabel` | `str` | `None` | **X轴标签**<br>通常设为 "Sample Index" |
| `ylabel` | `str` | `None` | **Y轴标签**<br>通常设为 "Sample Index" |
| `title` | `str` | `None` | **图表标题**<br>若为 `None`，自动生成包含样本数的默认标题 |
| `save_path` | `str` | `None` | **保存路径**<br>建议保存为 `.png` 或 `.pdf` |
| `show` | `bool` | `True` | **显示窗口**<br>是否弹出绘图窗口 |

### 5.3 plot_metric_line 参数说明

绘制多轮实验的性能变化折线图（Trace Plot），适用于展示算法的稳定性或收敛趋势。

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`results_list`** | **`List[Dict]`** | **必填** | **实验结果列表**<br>通常是 `evaluation_batch` 的输出结果，每个元素包含单次实验的各类指标得分 |
| `metrics` | `Union[List, str]` | `'ACC'` | **展示指标**<br>需要绘制的指标名称（如 `'ACC'`, `'NMI'`），支持单个字符串或字符串列表 |
| `xlabel` | `str` | `None` | **X轴标签**<br>默认为空，建议设为 "Run ID" 或 "Experiment ID" |
| `ylabel` | `str` | `None` | **Y轴标签**<br>默认为空，建议设为 "Score" |
| `title` | `str` | `None` | **图表标题**<br>若为 `None`，自动生成默认标题 |
| `save_path` | `str` | `None` | **保存路径**<br>自动去除白边并保存 |
| `show` | `bool` | `True` | **显示窗口**<br>是否弹出绘图窗口 |

### 5.4 plot_parameter_sensitivity 参数说明

基于网格搜索结果绘制单参数敏感性分析图，支持“控制变量法”逻辑，自动筛选特定背景参数下的性能曲线。

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`csv_file`** | **`str`** | **必填** | **数据源文件**<br>网格搜索生成的汇总 CSV 文件路径 |
| **`target_param`** | **`str`** | **必填** | **目标参数 (X轴)**<br>需要分析敏感性的参数名称（如 `'t'`, `'k'`） |
| `metric` | `str` | `'NMI'` | **评价指标 (Y轴)**<br>用于衡量性能的指标列名 |
| `fixed_params` | `Dict` | `None` | **固定背景参数**<br>字典格式（如 `{'k': 5}`）<br>若未指定，函数会自动寻找全局最优结果对应的参数作为固定背景 |
| `method_name` | `str` | `None` | **算法过滤器**<br>若 CSV 包含多种算法，需指定具体算法名称（如 `'mcla'`）以避免数据混淆 |
| `save_path` | `str` | `None` | **保存路径**<br>保存分析结果图 |
| `show` | `bool` | `True` | **显示窗口**<br>是否弹出绘图窗口 |

</details>

## <span id="pipelines">🚀 6. 流水线 (pce.pipelines)</span>

**自动化工作流模块，** 提供高层级的批处理接口，能够自动扫描数据集目录、串联“生成-集成-评估”全流程，适合大规模对比实验与结果复现。

<details>
<summary><strong>🔽 点击查看详细参数列表 (Click to expand)</strong></summary>

### 6.1 consensus_batch 参数说明

全自动集成流水线接口，支持目录扫描、基聚类自动生成（如果缺失）、集成运算、指标评估及结果保存（支持 CSV/Excel/MAT）。

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`input_dir`** | `str` | **必填** | **输入数据集目录**<br>代码会自动扫描该目录下所有的 `.mat` 文件进行处理 |
| `output_dir` | `str` | `None` | **结果输出目录**<br>如果为 `None`，默认使用输入目录作为输出根目录，并在其下创建算法对应的子文件夹 |
| `save_format` | `str` | `"csv"` | **结果保存格式**<br>支持 `'csv'`, `'xlsx'`, `'mat'`。需要确保 `pce.io` 中存在对应的 `save_` 函数 |
| `consensus_method` | `str` | `'cspa'` | **集成算法名称**<br>对应 `pce.consensus` 模块中的函数名，如 `'cspa'`, `'mcla'`, `'hgpa'` |
| `generator_method` | `str` | `'cdkmeans'` | **生成器名称**<br>当 `.mat` 文件中仅包含原始特征 `X` 时，使用此算法生成基聚类，如 `'cdkmeans'`, `'litekmeans'` |
| `nBase` | `int` | `20` | **基聚类数量/切片大小**<br>单次集成实验使用的基聚类器数量。若需现场生成 BPs，此参数也作为生成器的目标列数 |
| `seed` | `int` | `2024` | **随机种子**<br>用于控制基聚类生成和集成算法内部的随机性，保证结果可复现 |
| `maxiter` | `int` | `100` | **最大迭代次数**<br>仅在调用基聚类生成器（如 K-Means）时生效 |
| `replicates` | `int` | `1` | **生成器重复次数**<br>仅在调用基聚类生成器时生效，表示每次聚类尝试的重复运行次数 |
| `nRepeat` | `int` | `10` | **实验重复次数**<br>流水线将循环运行 `nRepeat` 次独立实验以评估算法稳定性 |
| `overwrite` | `bool` | `False` | **覆盖开关**<br>若为 `False` 且检测到输出文件已存在，则自动跳过该数据集。若为 `True` 则强制覆盖 |

</details>

## <span id="grid">🎛️ 7. 网格搜索 (pce.grid)</span>

**实验调优模块，** 专为科研设计的自动化参数搜索工具，能够生成参数空间的笛卡尔积，智能剔除冗余组合（Pruning），并自动记录详细的实验日志与汇总报表。

<details>
<summary><strong>🔽 点击查看详细参数列表 (Click to expand)</strong></summary>

### 7.1 GridSearcher 类说明

#### 7.1.1 初始化 (init)

初始化网格搜索器，设定输入输出路径及基础配置。

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`input_dir`** | **`str`** | **必填** | **输入数据集目录**<br>包含待实验 `.mat` 文件的文件夹路径，支持直接读取包含 `BPs` 的文件（跳过生成步骤）或仅包含 `X` 的文件（自动调用生成器） |
| **`output_dir`** | **`str`** | **必填** | **结果输出根目录**<br>所有实验结果、日志、JSON 配置将按数据集名称分类保存到此目录下 |

#### 7.1.2 运行 (run)

执行网格搜索任务，自动遍历参数组合并调用集成算法。

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`param_grid`** | **`Dict[str, List]`** | **必填** | **网格搜索参数空间**<br>字典的值必须是列表（如 `{'t': [0.5, 0.7]}`），程序会自动生成所有组合并根据算法类型智能清洗无效参数（如运行 `mcla` 时自动忽略只属于 `cspa` 的参数） |
| `fixed_params` | `Dict[str, Any]` | `None` | **固定参数配置**<br>所有实验共用的静态参数（如 `{'nBase': 20}`），如果 `param_grid` 中出现了同名参数，优先使用网格中的动态值 |

</details>

## <span id="utils">🛠️ 8. 工具模块 (pce.utils)</span>

**工具模块，** 用于开发调试与用户引导，支持打印算法的详细参数签名，并明确区分固定配置参数（Fixed）与可用于网格搜索的超参数（Hyperparameter）。

<details>
<summary><strong>🔽 点击查看详细参数列表 (Click to expand)</strong></summary>

### 8.1 show_function_params 参数说明

打印指定函数的参数列表，并智能区分超参数（Hyper）与固定配置参数（Fixed），辅助构建网格搜索配置。

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`method_name`** | **`str`** | **必填** | **目标函数名称**<br>例如 `'cspa'`, `'cdkmeans'` 等，需确保该名称在指定的 `module_type` 中存在 |
| `module_type` | `str` | `'consensus'` | **所属模块名称**<br>指定在哪个模块中查找函数，支持的选项映射：`'io'`, `'generators'`, `'consensus'`, `'metrics'`, `'pipelines'`, `'grid'`, `'analysis'` |

</details>

---

## <span id="roadmap">🎯 项目规划 (Roadmap)</span>

### ✅ 已完成特性 (Implemented Features)

**1. 基础架构与 IO (Infrastructure & IO)**
- [x] **MATLAB 深度兼容**: 完美支持 `.mat` v7.3 格式读取，自动处理 1-based/0-based 索引转换
- [x] **多格式导出**: 支持将实验结果导出为 CSV、Excel (自动保留小数位格式) 及 MAT 文件
- [x] **开发辅助**: 提供 `utils` 模块，支持查看算法参数签名及网格搜索兼容性

**2. 核心算法 (Core Algorithms)**
- [x] **基聚类生成器**: 支持 `LiteKMeans` (高速实现) 与 `CDK-Means` (约束性差异化生成)
- [x] **集成共识算法**: 实现了 `CSPA` (基于相似度)、`MCLA` (基于元聚类) 及 `HGPA` (基于超图) 等经典算法
- [x] **评估指标**: 内置 NMI, ARI, ACC, Purity, F-Score 等 14 种常用聚类指标

**3. 可视化分析 (Visualization & Analysis)**
- [x] **降维可视化**: 集成 t-SNE 与 PCA，支持原始数据及聚类结果的 2D 散点图绘制
- [x] **共协矩阵热力图**: 支持绘制排序后的共协矩阵 (Co-association Matrix) 以观察集成一致性
- [x] **性能分析**: 支持绘制多轮实验的指标折线图 (Trace Plot) 及超参数敏感性分析图

**4. 自动化与实验 (Automation)**
- [x] **批处理流水线**: `consensus_batch` 支持目录级扫描，一键完成“生成-集成-评估-保存”全流程
- [x] **智能网格搜索**: `GridSearcher` 支持参数笛卡尔积生成，具备**自动剪枝 (Pruning)** 功能，并自动记录详细日志

### 🚧 开发计划 (Future Plans)

**1. 算法扩展 (Algorithm Expansion)**
- [ ] **更多共识算法**: 计划新增 EAC (Evidence Accumulation), NMF-based 等集成策略
- [ ] **更多生成策略**: 计划新增基于随机投影或数据采样的基聚类生成器

**2. 工程化与发布 (Engineering)**
- [ ] **文档完善**: 部署 ReadTheDocs 在线文档与详细 API 索引
- [ ] **PyPI 发布**: 完成 v1.0.0 正式版发布流程

---

## 目前支持的功能

1. io

   - [x] load_mat.py（原始数据 / 基聚类）
   - [x] save_base.py（保存基聚类）
     - [x] save_base_mat.py（保存mat基聚类）
   - [x] save_results.py（保存结果）
     - [x] save_results_csv.py（保存csv结果）
     - [x] save_results_xlsx.py（保存xlsx结果）
     - [x] save_results_mat.py（保存mat结果）

2. 基聚类生成(generators)

   - [x] litekmeans.py（litekmeans生成基聚类）
   - [x] cdkmeans.py（cdkmeans生成基聚类）

3. 集成算法(consensus)

   - [x] cspa.py（cspa集成算法）
   - [x] mcla.py（mcla集成算法）
   - [x] hgpa.py（hgpa集成算法）

4. 评估指标(metrics)

   - [x] evaluation_single.py（评估单轮指标）
   - [x] evaluation_batch.py（评估多轮指标）

5. 分析(analysis)
    - [x] plot.py（绘图）

6. 流水线(pipelines)

    - [x] consensus_batch.py（集成流水线）

7. 网格搜索(grid)

    - [x] grid_search.py（网格搜索）
   
8. utils

   - [x] show_function_params.py（显示函数参数）

   

   




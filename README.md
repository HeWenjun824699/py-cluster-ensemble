# py-cluster-ensemble (PCE)

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue)](https://pypi.org/project/py-cluster-ensemble/) [![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A Python toolkit for Cluster Ensembles generation, consensus, and visualization.** 

`py-cluster-ensemble` 是一个模块化的 Python 聚类集成工具包，旨在协助研究人员从 MATLAB 迁移至 Python 环境。它提供了从基聚类生成、集成共识到结果评估和可视化的完整流水线，并完美兼容 `.mat` 数据格式。 

---

## 📋 目录 (Table of Contents) 

- [安装 (Installation)](#install) 
- [快速开始 (Quick Start)](#quickstart)
- [核心模块 API (API Reference)](#api_reference) 
  - [1. 输入输出 (IO)](#io)    
  - [2. 基聚类生成器 (Generators)](#generators) 
  - [3. 集成算法 (Consensus)](#consensus) 
  - [4. 评估指标 (Metrics)](#metrics) 
  - [5. 流水线 (Pipelines)](#pipelines) 
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

---


## <span id="api_reference">📚 核心模块 API (API Reference)</span>

## <span id="io">📂 1. 输入输出 (pce.io)</span>

<details>
<summary><strong>🔽 点击查看详细参数列表 (Click to expand)</strong></summary>

### load_mat 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`file_path`** | **`Union[str, Path]`** | **必填** | **输入 .mat 文件的路径**<br>支持字符串或 `pathlib.Path` 对象。代码会自动识别并处理标准 MATLAB 格式以及 v7.3 (HDF5) 格式 |
| `ensure_x_float` | `bool` | `True` | **强制转换特征矩阵为浮点数**<br>如果为 `True`，会将读取到的特征矩阵 `X` 转换为 `np.float64` 类型。这对大多数聚类算法（如 K-Means）的数值计算稳定性至关重要 |
| `flatten_y` | `bool` | `True` | **展平标签向量**<br>如果为 `True`，会将读取到的标签 `Y` 从二维列向量 `(n_samples, 1)` 展平为一维数组 `(n_samples,)`。这符合 Scikit-learn 等 Python 机器学习库的标准输入格式 |

### load_mat 返回值说明 (Returns)

| 变量名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **`X`** | `Optional[np.ndarray]` | **特征矩阵**<br>形状为 `(n_samples, n_features)`<br>代码会自动尝试匹配常见的变量名（如 `'X'`, `'data'`, `'fea'`, `'features'` 等）。如果未找到特征数据，函数将抛出 `IOError` |
| **`Y`** | `Optional[np.ndarray]` | **标签向量**<br>形状通常为 `(n_samples,)`（当 `flatten_y=True` 时）<br>代码会自动尝试匹配常见的变量名（如 `'Y'`, `'label'`, `'gnd'` 等）<br> **特殊处理**：如果读取到的标签是浮点型整数（如 `1.0`, `2.0`），代码会自动将其安全转换为 `np.int64` 类型 |

### save_csv 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`data`** | **`List[Dict]`** | **必填** | **评估结果数据**<br>通常是包含多轮实验结果的字典列表。列表中的每个字典代表一行数据，字典的 Key 将成为 CSV 的列名 |
| **`output_path`** | **`str`** | **必填** | **输出路径**<br>支持智能识别：<br>**1. 目录路径** (以 `/` 或 `\` 结尾，或已存在的文件夹)：文件将保存到该目录下，文件名由 `default_name` 指定<br>**2. 文件路径** (如 `output/res.csv`)：直接保存为该文件，代码会自动创建不存在的父目录 |
| `default_name` | `str` | `"result.csv"` | **默认文件名**<br>仅当 `output_path` 被判定为目录时使用 |
| `add_summary` | `bool` | `True` | **是否追加统计摘要**<br>如果为 `True`，会在原始数据后插入一个**空行**，然后计算并追加所有数值列的 **均值 (Mean)** 和 **标准差 (Std)** |
| `float_format` | `str` | `"%.4f"` | **浮点数格式控制**<br>指定写入 CSV 时浮点数的精度。默认保留 4 位小数 |

### save_xlsx 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`data`** | **`List[Dict]`** | **必填** | **评估结果数据**<br>通常是包含多轮实验结果的字典列表。列表中的每个字典代表一行数据，字典的 Key 将成为 Excel 的列名 |
| **`output_path`** | **`str`** | **必填** | **输出路径**<br>支持智能识别：<br>**1. 目录路径** (以 `/` 或 `\` 结尾，或已存在的文件夹)：文件将保存到该目录下，文件名由 `default_name` 指定<br>**2. 文件路径** (如 `output/res.xlsx`)：直接保存为该文件。代码会自动创建不存在的父目录，并强制后缀为 `.xlsx` |
| `default_name` | `str` | `"result.xlsx"` | **默认文件名**<br>仅当 `output_path` 被判定为目录时使用 |
| `add_summary` | `bool` | `True` | **是否追加统计摘要**<br>如果为 `True`，会在原始数据后插入一个**空行**，然后计算并追加所有数值列的 **均值 (Mean)** 和 **标准差 (Std)** |
| `excel_format` | `str` | `"0.0000"` | **Excel 数值格式控制**<br>指定 Excel 单元格的数字显示格式字符串。例如 `"0.0000"` 表示显示 4 位小数<br>相比 CSV，此方式能保持单元格为**数值类型**（便于在 Excel 中进行求和等后续计算），而非纯文本 |

### save_mat 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| **`data`** | **`List[Dict]`** | **必填** | **评估结果数据**<br>通常是包含多轮实验结果的字典列表。函数会将字典中的数值提取并转换为 NumPy 矩阵保存到 `.mat` 文件变量 `result` 中 |
| **`output_path**`** | **`str`** | **必填** | **输出路径**<br>支持智能识别：<br>**1. 目录路径** (以 `/` 或 `\` 结尾，或已存在的文件夹)：文件将保存到该目录下，文件名由 `default_name` 指定<br>**2. 文件路径** (如 `output/res.mat`)：直接保存为该文件。代码会自动创建不存在的父目录，并强制后缀为 `.mat` |
| `default_name` | `str` | `"result.mat"` | **默认文件名**<br>仅当 `output_path` 被判定为目录时使用 |
| `add_summary` | `bool` | `True` | **是否追加统计摘要**<br>如果为 `True`，会计算所有结果的 **均值 (Mean)** 和 **标准差 (Std)**，并分别以变量名 `result_summary` 和 `result_summary_std` 保存到 `.mat` 文件中 |

</details>

## <span id="generators">⚙️ 2. 基聚类生成器 (pce.generators)</span>

<details>
<summary><strong>🔽 点击查看详细参数列表 (Click to expand)</strong></summary>

### litekmeans 参数说明

| 参数名       | 类型             | 默认值   | 说明                                                         |
| :----------- | :--------------- | :------- | :----------------------------------------------------------- |
| **`X`**      | **`np.ndarray`** | **必填** | **输入特征矩阵**<br>形状通常为 `(n_samples, n_features)`, 算法将在此数据上运行 KMeans |
| **`Y`**      | **`np.ndarray`** | **必填** | **真实标签向量**<br>形状通常为 `(n_samples,)` 或 `(n_samples, 1)` <br>**用途：**代码内部使用 len(np.unique(Y)) 来确定数据集的真实类别数 nCluster，进而计算随机 K 值的范围 [minCluster, maxCluster] |
| `nBase`      | `int`            | `200`    | 生成基聚类（Base Partitions）的数量，即结果矩阵的列数        |
| `seed`       | `int`            | `2024`   | 随机种子，用于控制 K 值的随机选择和算法初始化，保证结果可复现 |
| `maxiter`    | `int`            | `100`    | 聚类算法（LiteKMeans）的最大迭代次数                         |
| `replicates` | `int`            | `1`      | 每次聚类尝试运行的重复次数，算法会返回其中目标函数最优的一次结果 |

### litekmeans 返回值说明(Returns)

| 变量名 | 类型          | 说明                                                                                                                     |
|:------|:-------------|:-------------------------------------------------------------------------------------------------------------------------|
| `BPs` | `np.ndarray` | **基聚类矩阵 (Base Partitions)**<br>形状为 (n_samples, nBase)。每一列代表一次 KMeans 聚类的结果标签（注意：代码中已对标签进行了 +1 处理，适应 MATLAB 风格索引，或便于区分）   |
| `Y`   | `np.ndarray` | **真实标签**<br>原样返回传入的 Y，便于后续流程直接调用                                                                                             |

### cdkmeans 参数说明

| 参数名       | 类型             | 默认值   | 说明                                                         |
| :----------- | :--------------- | :------- | :----------------------------------------------------------- |
| **`X`**      | **`np.ndarray`** | **必填** | **输入特征矩阵**<br>形状通常为` (n_samples, n_features)`, 算法将在此数据上运行 KMeans 初始化及 CDKM 优化 |
| **`Y`**      | **`np.ndarray`** | **必填** | **真实标签向量**<br>形状通常为 `(n_samples,)` 或 `(n_samples, 1)` <br>**用途：**代码内部使用 len(np.unique(Y)) 来确定数据集的真实类别数 nCluster，进而计算随机 K 值的范围 [minCluster, maxCluster] |
| `nBase`      | `int`            | `200`    | 生成基聚类（Base Partitions）的数量，即结果矩阵的列数        |
| `seed`       | `int`            | `2024`   | 随机种子，用于控制 K 值的随机选择和算法初始化，保证结果可复现 |
| `maxiter`    | `int`            | `100`    | 聚类算法（LiteKMeans）的最大迭代次数                         |
| `replicates` | `int`            | `1`      | 每次聚类尝试运行的重复次数，算法会返回其中目标函数最优的一次结果 |

### cdkmeans 返回值说明(Returns)

| 变量名 | 类型          | 说明                                                                                                                     |
|:------|:-------------|:-------------------------------------------------------------------------------------------------------------------------|
| `BPs` | `np.ndarray` | **基聚类矩阵 (Base Partitions)**<br>形状为 (n_samples, nBase)。每一列代表一次 KMeans 聚类的结果标签（注意：代码中已对标签进行了 +1 处理，适应 MATLAB 风格索引，或便于区分）   |
| `Y`   | `np.ndarray` | **真实标签**<br>原样返回传入的 Y，便于后续流程直接调用                                                                                             |

<br>

### litekmeans_old 参数说明

| 参数名           | 类型      | 默认值         | 说明                                                         |
| :-------------- | :-------- |:------------| :----------------------------------------------------------- |
| **`file_path`** | **`str`** | **必填**     | **输入数据文件的完整路径（支持 `.mat` 格式，兼容 v7.3）**    |
| `output_path`   | `str`     | `None`      | 输出 `.mat` 文件的保存路径（包含文件名）<br>如果为 `None`，默认保存到输入文件同目录下，文件名为 `[原文件名]_LKM[nBase].mat` |
| `nBase`         | `int`     | `200`       | 生成基聚类（Base Partitions）的数量，即结果矩阵的列数        |
| `seed`          | `int`     | `2024`      | 随机种子，用于控制 K 值的随机选择和算法初始化，保证结果可复现 |
| `maxiter`       | `int`     | `100`       | 聚类算法（LiteKMeans）的最大迭代次数                         |
| `replicates`    | `int`     | `1`         | 每次聚类尝试运行的重复次数，算法会返回其中目标函数最优的一次结果 |

### cdkmeans_old 参数说明

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

<details>
<summary><strong>🔽 点击查看详细参数列表 (Click to expand)</strong></summary>

### cspa 参数说明

| 参数名     | 类型      | 默认值         | 说明                                                                                                                                                       |
|:----------| :-------- |:------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`BPs`** | **`np.ndarray`** | **必填** | **基聚类矩阵 (Base Partitions)**<br>形状通常为 `(n_samples, n_total_clusterings)`<br>每一列代表一个基聚类器的结果, 代码内部会自动检测并处理 MATLAB 风格的 1-based 索引（将其转换为 Python 的 0-based 索引） |
| **`Y`**   | **`np.ndarray`** | **必填** | **真实标签向量**<br>形状为 `(n_samples,)` 或 `(n_samples, 1)` <br>**用途：**代码内部使用 `len(np.unique(Y))` 来确定最终集成聚类的目标类别数 `K(nCluster)`     |
| `nBase`   | `int`     | `20`        | **单次实验**使用的基聚类数量（切片大小）<br>例如：池中共有 200 个基聚类，设为 20 表示每次实验只使用其中 20 个来进行集成                                                                                   |
| `nRepeat` | `int`     | `10`        | 实验重复次数，程序会进行 `nRepeat` 次独立实验，所需的基聚类总列数 = `nBase` × `nRepeat`                                                                                            |
| `seed`    | `int`     | `2024`      | 随机种子，用于控制 CSPA 内部谱聚类（Spectral Clustering）的初始化状态，保证可复现性                                                                                                   |

### cspa 返回值说明(Returns)

| 变量名        | 类型               | 说明                                                         |
| :------------ | :----------------- | :----------------------------------------------------------- |
| `labels_list` | `List[np.ndarray]` | **预测标签列表**<br>包含 `nRepeat` 个元素的列表，每个元素是一个形状为 `(n_samples,)` 的一维 NumPy 数组，代表某次实验的 CSPA 集成结果 |
| `Y`           | `np.ndarray`       | **真实标签**<br>原样返回传入的 `Y`，便于后续流程直接调用     |

### mcla 参数说明

| 参数名      | 类型      | 默认值         | 说明                                                        |
|:-----------| :-------- |:------------| :----------------------------------------------------------- |
| **`BPs`**  | **`np.ndarray`** | **必填** | **基聚类矩阵 (Base Partitions)**<br>形状通常为 `(n_samples, n_total_clusterings)`<br>每一列代表一个基聚类器的结果, 代码内部会自动检测并处理 MATLAB 风格的 1-based 索引（将其转换为 Python 的 0-based 索引） |
| **`Y`**    | **`np.ndarray`** | **必填** | **真实标签向量**<br>形状为 `(n_samples,)` 或 `(n_samples, 1)` <br>**用途：**代码内部使用 `len(np.unique(Y))` 来确定最终集成聚类的目标类别数 `K(nCluster)` |
| `nBase`    | `int`     | `20`        | **单次实验**使用的基聚类数量（切片大小）<br>例如：池中共有 200 个基聚类，设为 20 表示每次实验只使用其中 20 个来进行集成    |
| `nRepeat`  | `int`     | `10`        | 实验重复次数，程序会进行 `nRepeat` 次独立实验，所需的基聚类总列数 = `nBase` × `nRepeat`             |
| `seed`     | `int`     | `2024`      | 随机种子，用于控制 MCLA 内部元聚类（Meta-Clustering）阶段的初始化状态（如谱聚类初始化），保证可复现性             |

### mcla 返回值说明(Returns)

| 变量名        | 类型               | 说明                                                         |
| :------------ | :----------------- | :----------------------------------------------------------- |
| `labels_list` | `List[np.ndarray]` | **预测标签列表**<br>包含 `nRepeat` 个元素的列表，每个元素是一个形状为 `(n_samples,)` 的一维 NumPy 数组，代表某次实验的 MCLA 集成结果 |
| `Y`           | `np.ndarray`       | **真实标签**<br>原样返回传入的 `Y`，便于后续流程直接调用     |

### hgpa 参数说明

| 参数名      | 类型      | 默认值         | 说明                                                        |
|:-----------| :-------- |:------------| :----------------------------------------------------------- |
| **`BPs`**  | **`np.ndarray`** | **必填** | **基聚类矩阵 (Base Partitions)**<br>形状通常为 `(n_samples, n_total_clusterings)`<br>每一列代表一个基聚类器的结果, 代码内部会自动检测并处理 MATLAB 风格的 1-based 索引（将其转换为 Python 的 0-based 索引） |
| **`Y`**    | **`np.ndarray`** | **必填** | **真实标签向量**<br>形状为 `(n_samples,)` 或 `(n_samples, 1)` <br>**用途：**代码内部使用 `len(np.unique(Y))` 来确定最终集成聚类的目标类别数 `K(nCluster)` |
| `nBase`    | `int`     | `20`        | **单次实验**使用的基聚类数量（切片大小）<br>例如：池中共有 200 个基聚类，设为 20 表示每次实验只使用其中 20 个来进行集成 |
| `nRepeat`  | `int`     | `10`        | 实验重复次数，程序会进行 `nRepeat` 次独立实验，所需的基聚类总列数 = `nBase` × `nRepeat` |
| `seed`     | `int`     | `2024`      | 随机种子，用于控制 HGPA 内部超图分割（Hypergraph Partitioning）阶段的初始化状态，保证可复现性 |

### hgpa 返回值说明(Returns)

| 变量名        | 类型               | 说明                                                         |
| :------------ | :----------------- | :----------------------------------------------------------- |
| `labels_list` | `List[np.ndarray]` | **预测标签列表**<br>包含 `nRepeat` 个元素的列表，每个元素是一个形状为 `(n_samples,)` 的一维 NumPy 数组，代表某次实验的 HGPA 集成结果 |
| `Y`           | `np.ndarray`       | **真实标签**<br>原样返回传入的 `Y`，便于后续流程直接调用     |

<br>

### cspa_old 参数说明

| 参数名           | 类型      | 默认值         | 说明                                                        |
| :-------------- | :-------- |:------------| :----------------------------------------------------------- |
| **`file_path`** | **`str`** | **必填**     | **输入数据文件的完整路径（`.mat` 格式）**<br>**文件内必须包含变量 `BPs` (基聚类矩阵) 和 `Y` (真实标签)** |
| `output_path`   | `str`     | `None`      | 输出 `.mat` 文件的保存路径 <br>如果为 `None`，默认在输入文件同级目录下新建 `CSPA_Results/[文件名]/` 保存结果 |
| `nBase`         | `int`     | `20`        | **单次实验**使用的基聚类数量（切片大小）<br>例如：池中共有 200 个基聚类，设为 20 表示每次实验只使用其中 20 个来进行集成 |
| `nRepeat`       | `int`     | `10`        | 实验重复次数，程序会进行 `nRepeat` 次独立实验，所需的基聚类总列数 = `nBase` × `nRepeat` |
| `seed`          | `int`     | `2024`      | 随机种子，用于控制 CSPA 内部谱聚类（Spectral Clustering）的初始化状态，保证可复现性 |

### mcla_old 参数说明

| 参数名           | 类型      | 默认值         | 说明                                                        |
| :-------------- | :-------- |:------------| :----------------------------------------------------------- |
| **`file_path`** | **`str`** | **必填**     | **输入数据文件的完整路径（`.mat` 格式）**<br>**文件内必须包含变量 `BPs` (基聚类矩阵) 和 `Y` (真实标签)**  |
| `output_path`   | `str`     | `None`      | 输出 `.mat` 文件的保存路径<br>如果为 `None`，默认在输入文件同级目录下新建 `MCLA_Results/[文件名]/` 保存结果 |
| `nBase`         | `int`     | `20`        | **单次实验**使用的基聚类数量（切片大小）<br>例如：池中共有 200 个基聚类，设为 20 表示每次实验只使用其中 20 个来进行集成    |
| `nRepeat`       | `int`     | `10`        | 实验重复次数，程序会进行 `nRepeat` 次独立实验，所需的基聚类总列数 = `nBase` × `nRepeat`             |
| `seed`          | `int`     | `2024`      | 随机种子，用于控制 MCLA 内部元聚类（Meta-Clustering）阶段的初始化状态（如谱聚类初始化），保证可复现性             |

### hgpa_old 参数说明

| 参数名           | 类型      | 默认值         | 说明                                                        |
| :-------------- | :-------- |:------------| :----------------------------------------------------------- |
| **`file_path`** | **`str`** | **必填**     | **输入数据文件的完整路径（`.mat` 格式）**<br>**文件内必须包含变量 `BPs` (基聚类矩阵) 和 `Y` (真实标签)** |
| `output_path`   | `str`     | `None`      | 输出 `.mat` 文件的保存路径<br>如果为 `None`，默认在输入文件同级目录下新建 `HGPA_Results/[文件名]/` 保存结果 |
| `nBase`         | `int`     | `20`        | **单次实验**使用的基聚类数量（切片大小）<br>例如：池中共有 200 个基聚类，设为 20 表示每次实验只使用其中 20 个来进行集成 |
| `nRepeat`       | `int`     | `10`        | 实验重复次数，程序会进行 `nRepeat` 次独立实验，所需的基聚类总列数 = `nBase` × `nRepeat` |
| `seed`          | `int`     | `2024`      | 随机种子，用于控制 HGPA 内部超图分割（Hypergraph Partitioning）阶段的初始化状态，保证可复现性 |

</details>

## <span id="metrics">📊 4. 评估指标 (pce.metrics)</span>

<details>
<summary><strong>🔽 点击查看详细参数列表 (Click to expand)</strong></summary>

### evaluation_single 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | ---- |
| **`y`** | **`np.ndarray`** | **必填** | **预测标签向量**<br>聚类算法输出的预测结果，代码内部会自动展平为一维数组 |
| **`Y`** | **`np.ndarray`** | **必填** | **真实标签向量**<br>数据集的真实标签 (Ground Truth)，用于评估聚类性能，代码内部会自动展平为一维数组 |

### evaluation_single 返回值说明 (Returns)

| 变量名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **`res`** | `List[float]` | **评估指标列表**<br>包含14个浮点数的列表，顺序依次为：<br>`[ACC, NMI, Purity, AR, RI, MI, HI, F-Score, Precision, Recall, Entropy, SDCS, RME, Bal]` |

### evaluation_batch 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | ---- |
| **`labels`** | **`List[np.ndarray]`** | **必填** | **预测标签列表**<br>包含多个预测结果的列表，列表中的每个元素都是一个形状为 `(n_samples,)` 的一维数组（例如多次实验的聚类结果），函数将遍历此列表逐一评估 |
| **`Y`** | **`np.ndarray`** | **必填** | **真实标签向量**<br>数据集的真实标签 (Ground Truth)，用于评估聚类性能，代码内部会自动展平为一维数组 |

### evaluation_batch 返回值说明 (Returns)

| 变量名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **`res_list`** | `List[Dict]` | **评估结果列表**<br>列表中的每个元素都是一个字典，对应 `labels` 中每一次预测的评估结果。字典包含以下 14 个 Key：<br>`['ACC', 'NMI', 'Purity', 'AR', 'RI', 'MI', 'HI', 'F-Score', 'Precision', 'Recall', 'Entropy', 'SDCS', 'RME', 'Bal']` |

</details>

### <span id="pipelines">🚀 5. 流水线 (pce.pipelines)</span>

<details>
<summary><strong>🔽 点击查看详细参数列表 (Click to expand)</strong></summary>

### consensus_batch 参数说明

### consensus_batch 参数说明

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

## <span id="roadmap">🗺 项目规划 (Roadmap)</span>

- [x] 基础架构: IO, Generators, Consensus, Metrics 模块解耦
- [x] 兼容性: 完美支持 MATLAB .mat v7.3 及 1-based 索引自动修复
- [x] 流水线: consensus_batch 自动化批处理与断点跳过
- [x] 报表: 支持生成带格式的 Excel 报表
- [ ] 可视化: 共识矩阵 (Consensus Matrix) 热力图
- [ ] 发布: PyPI 正式发布

<br>

**方案：**

1. 所有代码转python

2. 分模块

   - 基聚类生成
   - 集成方法
   - 指标
   - 可视化

3. 调用demo

   使用工具包基于`litekmeans`和`CDKM_fast`生成基聚类

   ~~~
   import pce
   
   isolet_filepath = 'data/isolet_uni_1560n_617d_2c.mat'
   pce.generation.cdkmeans(isolet_filepath, output_path='data/CDKM200')
   ~~~



<br>

**Paper demo:**

WEFE A Python Library for Measuring and Mitigating Bias in Word Embeddings-JMLR-2025

NUBO A Transparent Python Package for Bayesian Optimization-JSS-2025


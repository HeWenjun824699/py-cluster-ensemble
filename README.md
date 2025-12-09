## generation

### litekmeans 参数说明

| 参数名          | 类型      | 默认值   | 说明                                                         |
| :-------------- | :-------- | :------- | :----------------------------------------------------------- |
| **`file_path`** | **`str`** | **必填** | **输入数据文件的完整路径（支持 `.mat` 格式，兼容 v7.3）**    |
| `output_path`   | `str`     | `None`   | 输出 `.mat` 文件的保存路径（包含文件名）<br>如果为 `None`，默认保存到输入文件同目录下，文件名为 `[原文件名]_LKM[nBase].mat` |
| `nBase`         | `int`     | `200`    | 生成基聚类（Base Partitions）的数量，即结果矩阵的列数        |
| `seed`          | `int`     | `2024`   | 随机种子，用于控制 K 值的随机选择和算法初始化，保证结果可复现 |
| `maxiter`       | `int`     | `100`    | 聚类算法（LiteKMeans）的最大迭代次数                         |
| `replicates`    | `int`     | `1`      | 每次聚类尝试运行的重复次数，算法会返回其中目标函数最优的一次结果 |

### cdkmeans 参数说明

| 参数名          | 类型      | 默认值   | 说明                                                         |
| :-------------- | :-------- | :------- | :----------------------------------------------------------- |
| **`file_path`** | **`str`** | **必填** | **输入数据文件的完整路径（支持 `.mat` 格式，兼容 v7.3）**    |
| `output_path`   | `str`     | `None`   | 输出 `.mat` 文件的保存路径（包含文件名）<br>如果为 `None`，默认保存到输入文件同目录下，文件名为 `[原文件名]_CDKM[nBase].mat` |
| `nBase`         | `int`     | `200`    | 生成基聚类（Base Partitions）的数量，即结果矩阵的列数        |
| `seed`          | `int`     | `2024`   | 随机种子，用于控制 K 值的随机选择和算法初始化，保证结果可复现 |
| `maxiter`       | `int`     | `100`    | 聚类算法（LiteKMeans）的最大迭代次数                         |
| `replicates`    | `int`     | `1`      | 每次聚类尝试运行的重复次数，算法会返回其中目标函数最优的一次结果 |

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


## generation

### litekmeans参数说明



### cdkmeans参数说明

| 参数名          | 类型  | 默认值   | 说明                                                         |
| :-------------- | :---- | :------- | :----------------------------------------------------------- |
| **`file_path`** | `str` | **必填** | 输入数据文件的完整路径（支持 `.mat` 格式，兼容 v7.3）        |
| `output_path`   | `str` | `None`   | 输出 `.mat` 文件的保存路径（包含文件名）<br>如果为 `None`，默认保存到输入文件同目录下，文件名为 `[原名]_CDKM_[nBase].mat` |
| `nBase`         | `int` | `200`    | 生成基聚类（Base Partitions）的数量，即结果矩阵的列数        |
| `seed`          | `int` | `2024`   | 随机种子，用于控制 K 值的随机选择和算法初始化，保证结果可复现 |
| `maxiter`       | `int` | `100`    | 聚类算法（LiteKMeans）的最大迭代次数                         |
| `replicates`    | `int` | `1`      | 每次聚类尝试运行的重复次数，算法会返回其中目标函数最优的一次结果 |





核心卖点：

1. 整合算法便于对比实验？生成基聚类？评估？可视化？



方案：

1. 所有代码转python

2. 分阶段（生成→评估→选择→增强→融合→验证）

   基聚类生成API：基聚类 = 用户传参（随机种子，生成策略）

3. 搭建一个算法全流程跑通

4. 调用demo

   ~~~
   import censemble
   
   # 生成基聚类
   xxx = censemble.generate(参数)
   
   # 选择
   X, score = censemble.selection(xxx, model)
   
   # 增强
   X = censemble.enhanced(X, model)
   
   # 融合
   X = censemble.fusion(X, model)
   
   # 评估
   acc,nmi,ari = censemble.eval(X, model)
   
   # 可视化
   censemble.selection(xxx, method)   # 弹窗展示
   ~~~

5. 注册test pypi和pypi账号



Paper demo:

WEFE A Python Library for Measuring and Mitigating Bias in Word Embeddings-JMLR-2025

NUBO A Transparent Python Package for Bayesian Optimization-JSS-2025


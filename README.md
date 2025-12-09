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


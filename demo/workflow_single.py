import numpy as np

import pce

data_path = r"../data/isolet_uni_1560n_617d_2c.mat"

X, Y = pce.io.load_mat_X_Y(data_path)
print(X)
print(Y)

print("====================Generating...====================")
# BPs = pce.generators.litekmeans(X, Y)
BPs = pce.generators.cdkmeans(X, Y)
print(BPs)
print(Y)

print("====================Save Bases...====================")
pce.io.save_base_mat(BPs, Y, "F:\Projects\py-cluster-ensemble\output\single\isolet_uni_1560n_617d_2c_CDKM200.mat")

print("====================Consensus...====================")
nClusters = len(np.unique(Y))
# labels = pce.consensus.cspa(BPs, nClusters=nClusters)
# labels = pce.consensus.mcla(BPs, nClusters=nClusters)
labels, time_list = pce.consensus.hgpa(BPs, nClusters=nClusters)
print(labels)

print("====================Evaluating...====================")
res = pce.metrics.evaluation_batch(labels, Y, time_list)
print(res)

print("====================Saving...====================")
pce.io.save_results_csv(res, r"F:\Projects\py-cluster-ensemble\output\single\isolet_uni_1560n_617d_2c_CDKM200_HGPA.csv")

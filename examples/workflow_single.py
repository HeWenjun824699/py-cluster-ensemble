import numpy as np

import pce

data_path = r"./data/isolet_1560n_617d_2c.mat"

X, Y = pce.io.load_mat_X_Y(data_path)
print(X)
print(Y)

print("====================Generating...====================")
BPs = pce.generators.cdkmeans(X, Y)
print(BPs)
print(Y)

print("====================Save Bases...====================")
pce.io.save_base_mat(BPs, Y, r"./results/workflow/single/isolet_1560n_617d_2c_CDKM200.mat")

print("====================Consensus...====================")
nClusters = len(np.unique(Y))
labels_list, time_list = pce.consensus.cspa(BPs, nClusters=nClusters)
print(labels_list)

print("====================Evaluating...====================")
res = pce.metrics.evaluation_batch(labels_list, Y, time_list)
print(res)

print("====================Saving...====================")
pce.io.save_results_csv(res, r"./results/workflow/single/isolet_1560n_617d_2c_CDKM200_CSPA.csv")

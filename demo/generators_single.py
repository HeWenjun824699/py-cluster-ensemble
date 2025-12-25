import pce

data_path = r"./data/isolet_1560n_617d_2c.mat"

X, Y = pce.io.load_mat_X_Y(data_path)
print(X)
print(Y)

print("====================Generating...====================")

# Lite K-Means
base_output_path = r"./data/LKM200/isolet_uni_1560n_617d_2c_LDKM200.mat"
BPs = pce.generators.litekmeans(
    X=X,
    Y=Y,
    nClusters=None,
    nPartitions=200,
    seed=2026,
    maxiter=100,
    replicates=1
)

# # Coordinate Descent K-Means
# base_output_path = r"./data/CDKM200/isolet_uni_1560n_617d_2c_CDKM200.mat"
# BPs = pce.generators.cdkmeans(
#     X=X,
#     Y=Y,
#     nClusters=None,
#     nPartitions=200,
#     seed=2026,
#     maxiter=100,
#     replicates=1
# )

# # Random Subspace K-Means
# base_output_path = r"./data/RSKM200/isolet_uni_1560n_617d_2c_RSKM200.mat"
# BPs = pce.generators.rskmeans(
#     X=X,
#     Y=Y,
#     nClusters=None,
#     nPartitions=200,
#     subspace_ratio=0.5,
#     seed=2026,
#     maxiter=100,
#     replicates=1
# )

# # Random Partition K-Means
# base_output_path = r"./data/RPKM200/isolet_uni_1560n_617d_2c_RPKM200.mat"
# BPs = pce.generators.rpkmeans(
#     X=X,
#     Y=Y,
#     nClusters=None,
#     nPartitions=200,
#     projection_ratio=0.5,
#     seed=2026,
#     maxiter=100,
#     replicates=1
# )

# # BAG K-Means
# base_output_path = r"./data/BAGKM200/isolet_uni_1560n_617d_2c_BAGKM200.mat"
# BPs = pce.generators.bagkmeans(
#     X=X,
#     Y=Y,
#     nClusters=None,
#     nPartitions=200,
#     subsample_ratio=0.8,
#     seed=2026,
#     maxiter=100,
#     replicates=1
# )

# # Heterogeneous Clustering
# base_output_path = r"./data/HETCLU200/isolet_uni_1560n_617d_2c_HETCLU200.mat"
# BPs = pce.generators.hetero_clustering(
#     X=X,
#     Y=Y,
#     nClusters=None,
#     nPartitions=200,
#     seed=2026
# )

print(BPs)
print(Y)

print("====================Save Bases...====================")
pce.io.save_base_mat(BPs, Y, base_output_path)

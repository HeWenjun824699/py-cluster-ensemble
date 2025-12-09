import os
import math
import numpy as np
import scipy.io
import h5py

from .methods.litekmeans import litekmeans as litekmeans_methods


def extract_xy(file_path):
    """
    增强版 extractXY：支持标准 .mat 和 v7.3 格式 (HDF5)。
    """
    X = None
    Y = None

    # --- 尝试 1: 使用 scipy.io (适用于旧版/标准 .mat) ---
    try:
        data = scipy.io.loadmat(file_path)

        # 寻找 X
        for key in ['X', 'data', 'fea', 'features']:
            if key in data:
                X = data[key]
                break

        # 寻找 Y
        for key in ['Y', 'label', 'gnd', 'labels']:
            if key in data:
                Y = data[key]
                break

        if X is not None:
            X = X.astype(float)
        if Y is not None:
            Y = Y.flatten()

    except Exception as e:
        # 如果报错包含 v7.3 提示，或者是 NotImplementedError
        if 'v7.3' in str(e) or isinstance(e, NotImplementedError):
            print(f"Detected v7.3 mat file, switching to h5py for {file_path}...")

            # --- 尝试 2: 使用 h5py (适用于 v7.3 .mat) ---
            try:
                with h5py.File(file_path, 'r') as f:
                    # 寻找 X
                    for key in ['X', 'data', 'fea', 'features']:
                        if key in f:
                            # 【关键】MATLAB v7.3 读入后通常是转置的，需要 .T
                            X = np.array(f[key]).T
                            break

                    # 寻找 Y
                    for key in ['Y', 'label', 'gnd', 'labels']:
                        if key in f:
                            Y = np.array(f[key]).T
                            break

                if X is not None:
                    X = X.astype(float)
                if Y is not None:
                    Y = Y.flatten()

            except Exception as h5_e:
                print(f"h5py failed as well: {h5_e}")
                return None, None
        else:
            print(f"Error loading {file_path}: {e}")
            return None, None

    if X is None or Y is None:
        print(f"Could not find X or Y in {file_path}")
        return None, None

    return X, Y


def litekmeans(filepath, nBase=200, seed=2024, maxiter=100, replicates=1):
    """
    主函数：批量生成基聚类 (Base Partitions)
    对应 MATLAB 脚本的主逻辑
    """
    file_name = os.path.basename(filepath)
    data_name = os.path.splitext(file_name)[0]

    # 如果输出目录不存在，创建它
    output_path = os.path.dirname(filepath)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created directory: {output_path}")

    # 提取数据
    try:
        X, Y = extract_xy(filepath)
        if X is None: return
    except Exception as e:
        print(f"Skipping {data_name}: {e}")
        return

    nSmp = X.shape[0]
    nCluster = len(np.unique(Y))

    # 计算 K 值范围 (minCluster, maxCluster)
    # 对应 MATLAB: min(nCluster, ceil(sqrt(nSmp)))
    sqrt_n = math.ceil(math.sqrt(nSmp))
    minCluster = min(nCluster, sqrt_n)
    maxCluster = max(nCluster, sqrt_n)

    # 构造输出文件名
    out_file_name = os.path.splitext(data_name)[0] + '_LKM_' + str(nBase) + '.mat'
    out_file_path = os.path.join(os.path.dirname(file_name), out_file_name)

    # --- 3. 生成基聚类 ---
    if not os.path.exists(out_file_path):
        print(f"Processing {data_name}...")

        BPs = np.zeros((nSmp, nBase), dtype=int)

        nRepeat = nBase

        # 初始化随机数生成器 (对应 MATLAB: seed = 2024; rng(seed))
        # 我们先生成 200 个随机种子，用于控制每一次循环
        rs = np.random.RandomState(seed)
        random_seeds = rs.randint(0, 1000001, size=nRepeat)

        for iRepeat in range(nRepeat):
            current_seed = random_seeds[iRepeat]

            # -------------------------------------------------
            # 步骤 A: 随机选择 K 值
            # -------------------------------------------------
            np.random.seed(current_seed)

            if minCluster == maxCluster:
                iCluster = minCluster
            else:
                iCluster = np.random.randint(minCluster, maxCluster + 1)

            # -------------------------------------------------
            # 步骤 B: 运行 LiteKMeans
            # -------------------------------------------------
            np.random.seed(current_seed)

            # 调用 litekmeans
            label = litekmeans_methods(X, iCluster, maxiter=maxiter, replicates=replicates)[0]

            BPs[:, iRepeat] = label

        # --- 4. 保存结果 ---
        scipy.io.savemat(out_file_path, {'BPs': BPs, 'Y': Y})
        print(f"{data_name} has been completed!")

    else:
        print(f"{data_name} already exists. Skipping.")


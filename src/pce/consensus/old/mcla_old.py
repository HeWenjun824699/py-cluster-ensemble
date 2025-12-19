import os
import time
import numpy as np
import scipy.io

from pce.consensus.methods.mcla_core import mcla_core
from pce.metrics.evaluation_single import evaluation_single


def mcla_old(file_path, output_path=None, nBase: int = 20, nRepeat: int = 10, seed: int = 2026):
    """
    MCLA (Meta-Clustering Algorithm) Wrapper.
    对应 MATLAB 脚本的主逻辑：批量读取 BPs，切片运行 MCLA，评估并保存结果。
    """
    file_name = os.path.basename(file_path)
    data_name = os.path.splitext(file_name)[0]

    # 1. 路径与文件名处理
    # 如果未指定输出路径，默认在输入文件同级目录下创建一个 MCLA_Results 文件夹
    if output_path is None:
        input_dir = os.path.dirname(file_path)
        # 结构: .../data_dir/MCLA_Results/data_name/
        output_path = os.path.join(input_dir, 'MCLA_Results', data_name)

    # 如果输出目录不存在，创建它
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created directory: {output_path}")

    # 构造输出文件名
    out_file_name = f"{data_name}_MCLA.mat"
    out_file_path = os.path.join(output_path, out_file_name)

    # 2. 检查结果是否已存在
    if os.path.exists(out_file_path):
        print(f"{data_name} already exists. Skipping.")
        return

    # 3. 提取数据 (加载 BPs 和 Y)
    try:
        mat_data = scipy.io.loadmat(file_path)

        if 'BPs' not in mat_data or 'Y' not in mat_data:
            print(f"Skipping {data_name}: 'BPs' or 'Y' not found in .mat file.")
            return

        BPs = mat_data['BPs']
        Y = mat_data['Y'].flatten()

        # 【关键】处理 MATLAB 的 1-based 索引
        # MCLA 核心算法通常也需要 0-based 索引来构建超图或矩阵
        if np.min(BPs) == 1:
            BPs = BPs - 1

        nSmp = BPs.shape[0]
        nTotalBase = BPs.shape[1]
        nCluster = len(np.unique(Y))

    except Exception as e:
        print(f"Error loading {data_name}: {e}")
        return

    # 4. 实验循环
    print(f"MCLA Processing {file_name} (Shape: {BPs.shape}, K={nCluster})...")

    # 准备结果容器
    results_list = []

    # 初始化随机数生成器
    rs = np.random.RandomState(seed)
    random_seeds = rs.randint(0, 1000001, size=nRepeat)

    for iRepeat in range(nRepeat):
        # -------------------------------------------------
        # 步骤 A: 切片 BPs
        # -------------------------------------------------
        start_idx = iRepeat * nBase
        end_idx = (iRepeat + 1) * nBase

        # 边界检查
        if start_idx >= nTotalBase:
            print(f"Warning: Not enough Base Partitions for repeat {iRepeat + 1}")
            break
        if end_idx > nTotalBase:
            end_idx = nTotalBase

        BPi = BPs[:, start_idx:end_idx]

        # -------------------------------------------------
        # 步骤 B: 运行 MCLA
        # -------------------------------------------------
        current_seed = random_seeds[iRepeat]

        t_start = time.time()

        try:
            # 调用核心算法
            # 注意：Python 实现通常直接接收 (n_samples, n_estimators)
            # 但 mcla_core (移植自 MATLAB) 期望 (n_clusterings, n_samples)
            label_pred = mcla_core(BPi.T, nCluster)
        except Exception as e:
            print(f"MCLA failed on repeat {iRepeat}: {e}")
            label_pred = np.zeros_like(Y)

        t_cost = time.time() - t_start

        # -------------------------------------------------
        # 步骤 C: 评估
        # -------------------------------------------------
        metrics = evaluation_single(label_pred, Y)

        # 保存单次结果: [NMI, ARI, ACC, ..., Time]
        row_result = metrics + [t_cost]
        results_list.append(row_result)

    # 5. 汇总与保存
    results_mat = np.array(results_list)

    # 计算均值和方差 (注意标准差使用 ddof=1 以匹配 MATLAB std)
    summary_mean = np.mean(results_mat, axis=0)
    summary_std = np.std(results_mat, axis=0, ddof=1)

    # 构造保存字典
    save_dict = {
        'MCLA_result': results_mat,
        'MCLA_result_summary': summary_mean,
        'MCLA_result_summary_std': summary_std
    }

    scipy.io.savemat(out_file_path, save_dict)
    print(f"{data_name} has been completed! Saved to: {out_file_path}")

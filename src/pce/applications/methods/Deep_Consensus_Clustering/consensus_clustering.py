import os
import pickle
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import correspond, worker_kmeans_dynamic
from ....consensus.dcc import dcc


def run_consensus_clustering(input_path, output_path, hidden_dims, k_min=3, k_max=10, **kwargs):
    """
    Args:
        input_path: 数据集根目录
        output_path: 输出根目录 (包含 representations 和 results)
        hidden_dims: list, 之前生成表示时使用的 hidden_dims 列表
        k_min, k_max: 聚类数范围
    """

    cfg = {
        'seed': 2026
    }
    cfg.update(kwargs)
    args = SimpleNamespace(**cfg)

    # 1. 路径准备
    rep_folder = os.path.join(output_path, "representations")
    res_folder = os.path.join(output_path, "results")
    pkl_folder = os.path.join(res_folder, "pkls")
    png_folder = os.path.join(res_folder, "pngs")
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    if not os.path.exists(pkl_folder):
        os.makedirs(pkl_folder)
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    # 2. 加载 Ground Truth (用于排序)
    data_file = os.path.join(input_path, 'data.pkl')
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    y = np.array(data[1])

    # 3. 初始化变量
    cdf = []
    areas = []
    consensus_bars = []

    # 确保 hidden_dims 是列表 (如果是 range 对象转为 list)
    hidden_dims_list = list(hidden_dims)
    n_estimators = len(hidden_dims_list)

    print(f"\nStarting Consensus Clustering k={k_min}-{k_max}, models={n_estimators}...")

    plt.figure()

    # 4. 主循环：遍历不同的 K 值
    for k in range(k_min, k_max):
        print(f"  Processing k={k}...")

        # --- A. 生成基聚类 (Base Clustering) ---
        # 这里的 i 对应 hidden_dims_list 中的每一个 hidden_dim
        clusters = [worker_kmeans_dynamic(rep_folder, k, i, seed=args.seed) for i in
                    tqdm(hidden_dims_list, desc=f"Base Clustering k={k}")]

        """
        旧逻辑:
        
        clusters = np.array(clusters)  # Shape: [num_models, num_samples]

        # --- B. 计算共识矩阵 (Consensus Matrix) ---
        # 改为单线程执行：避免多进程传输大数组时的序列化开销和死锁风险
        print(f"  Calculating Consensus Matrix for k={k}...")

        # 直接使用列表推导式 (List Comprehension) 配合 tqdm 显示进度
        m = [worker_m(clusters, i) for i in tqdm(range(clusters.shape[1]), desc="Building Matrix")]

        # 归一化：除以基聚类模型的数量 (即 hidden_dims 的个数)
        m = np.array(m, dtype=np.float32) / len(hidden_dims_list)

        # 保存原始共识矩阵
        with open(os.path.join(res_folder, 'pkls', f'm_{k}.pkl'), 'wb') as f:
            pickle.dump(m, f)

        # --- C. 最终聚类 (Final Clustering on Matrix) ---
        model = KMeans(n_clusters=k, random_state=args.seed)
        model.fit(m)
        res = np.array(model.labels_)

        # 根据死亡率排序 (re-labeling)
        res = correspond(res, y)
        """

        # 转换为 dcc_consensus 需要的格式 (n_samples, n_estimators)
        BPs = np.array(clusters).T

        # --- B & C. 替换为调用 dcc_consensus ---
        # 我们设置 nRepeat=1，因为这里的逻辑是针对特定 k 的一次确定性分析
        print(f"  Running DCC Consensus for k={k}...")
        labels_list, _, m = dcc(
            BPs=BPs,
            Y=None,
            nClusters=k,  # 当前循环的 k
            nBase=n_estimators,  # 基聚类数量
            nRepeat=1,
            seed=args.seed,
            return_matrix=True
        )

        # 获取结果 (因为 nRepeat=1, 取第一个)
        res = labels_list[0]

        # 保存原始共识矩阵 (保持原有逻辑)
        with open(os.path.join(res_folder, 'pkls', f'm_{k}.pkl'), 'wb') as f:
            pickle.dump(m, f)

        # --- 后处理：根据 Ground Truth 排序 (Re-labeling) ---
        res = correspond(res, y)

        s = sorted(res)
        with open(os.path.join(res_folder, 'pkls', f'consensus_cluster_{k}.pkl'), 'wb') as f:
            pickle.dump(res, f)

        # --- D. 计算统计指标 (CDF, PAC 等) ---
        # 计算簇的边界 (用于可视化)
        c_num = []
        for i in range(1, len(s)):
            if s[i] != s[i - 1]:
                c_num.append(i)
        c_num.append(len(s))

        # 对矩阵进行排序以便可视化 (可选，此处主要为了计算指标)
        m = m[:, res.argsort()]
        m = m[res.argsort(), :]

        # 计算 Consensus Value (用于柱状图)
        b = []
        for i in range(len(c_num)):
            if i == 0:
                area_sum = m[:c_num[0], :c_num[0]].sum()
                count = c_num[0] * c_num[0]
                b.append(area_sum / count if count > 0 else 0)
            else:
                area_sum = m[c_num[i - 1]:c_num[i], c_num[i - 1]:c_num[i]].sum()
                count = (c_num[i] - c_num[i - 1]) ** 2
                b.append(area_sum / count if count > 0 else 0)
        consensus_bars.append(b)

        # 计算 CDF
        consensus_value = m.ravel()
        hist, bin_edges = np.histogram(consensus_value, bins=100, range=(0, 1))
        # 修正直方图统计 (排除自身的对角线或微小误差，这里保留原逻辑)
        # hist[-1] -= m.shape[0]

        c = np.cumsum(hist / sum(hist))
        cdf.append(c)

        # 绘图: CDF 曲线
        width = (bin_edges[1] - bin_edges[0])
        plt.plot(bin_edges[1:] - width / 2, c, label=f'k={k}')

        # 计算 Area Under CDF
        # 这里使用简单的矩形近似或梯形公式
        delta_a = [h * width for h in c]  # 简化计算
        a = np.sum(delta_a)
        areas.append(a)

    # 5. 保存图表
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid()
    plt.legend()
    plt.xlabel('Consensus Index')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF of Consensus Values')
    plt.savefig(os.path.join(res_folder, 'pngs', 'cdf.png'), dpi=300)
    plt.close()

    # Delta K 图
    delta_k = []
    for i in range(len(areas)):
        if i == 0:
            delta_k.append(areas[0])  # 或者 0，视定义而定
        else:
            # Relative change calculation
            if areas[i - 1] != 0:
                delta_k.append((areas[i] - areas[i - 1]) / areas[i - 1])
            else:
                delta_k.append(0)

    plt.figure()
    k_range = range(k_min, k_max)
    plt.plot([i for i in k_range], delta_k)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Relative Change')
    plt.title('Relative Change in Area Under CDF')
    plt.xticks([i for i in k_range])
    plt.savefig(os.path.join(res_folder, 'pngs', 'delta.png'), dpi=300)
    plt.close()

    # Average Consensus 图
    plt.figure()
    index = 0
    i_s = []
    for i in range(len(consensus_bars)):
        k_val = k_range[i]
        b = consensus_bars[i]
        x = [j for j in range(index, index + len(b))]
        plt.bar(x, b, label=f'k={k_val}')
        i_s.append(index + len(b) * 0.5 - 0.5)
        index += len(b) + 2

    plt.xticks(ticks=i_s, labels=[f"k={i}" for i in k_range])
    plt.xlabel('Subphenotypes')
    plt.ylabel('Average Consensus Value')
    plt.title('Average Consensus Value')
    plt.savefig(os.path.join(res_folder, 'pngs', 'aver_con.png'), dpi=300)
    plt.close()

    print(f"Consensus Clustering Completed. Results saved to {res_folder}")

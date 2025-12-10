import os
import numpy as np
from .ints import ints


def wgraph(e, w=None, method=0, dataname=None):
    """
    Writes the graph file for METIS.
    Fixed version: Added missing loop for method 0 (CSPA graph).
    """
    if w is None:
        w = []

    # 1. 路径处理
    folder = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(folder, 'tmp')

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    if dataname is None:
        dataname = os.path.join(tmp_dir, 'graph')

    # 处理文件名冲突 (graph0, graph00, ...)
    base_dataname = dataname + str(method)
    dataname = base_dataname
    while os.path.exists(dataname):
        dataname = dataname + str(method)

    # 2. 预处理矩阵
    if method == 0 or method == 1:
        # 去除对角线元素 (自环)
        np.fill_diagonal(e, 0)

    # 确保转换为整数 (METIS 只能处理整数权重)
    e = ints(e)
    if method == 1 or method == 3:
        w = ints(w)

    try:
        # 使用 newline='\n' 确保换行符一致
        with open(dataname, 'w', newline='\n') as f:

            # ==========================================
            # 分支 A: 普通图 (CSPA 用这个, method=0)
            # ==========================================
            if method == 0 or method == 1:
                # 1. 写入 Header: Nodes Edges Format
                # Format=1 表示有边权重 (Edge Weights)
                # Format=11 表示有点权重和边权重
                n_edges = int(np.sum(e > 0) / 2)  # 无向图边数
                fmt_code = "1" if method == 0 else "11"

                f.write(f"{e.shape[0]} {n_edges} {fmt_code}\n")

                # 2. 【关键修复】写入具体数据
                # 遍历每个节点 (行)
                for i in range(e.shape[0]):
                    # 找到邻居 (e[i, j] > 0 的位置)
                    # np.where 返回的是 tuple, 取 [0]
                    neighbor_indices = np.where(e[i, :] > 0)[0]
                    weights = e[i, neighbor_indices]

                    line_parts = []

                    # 如果是 method 1，还需要先写入节点自身的权重
                    if method == 1:
                        if w is not None and len(w) > i:
                            line_parts.append(str(int(w[i])))
                        else:
                            line_parts.append("1")  # 默认权重

                    # 写入邻居和权重对: neighbor weight neighbor weight ...
                    # 注意：METIS 索引是 1-based，所以 idx 要 +1
                    for idx, weight in zip(neighbor_indices, weights):
                        line_parts.append(str(idx + 1))  # 邻居 ID (+1)
                        line_parts.append(str(int(weight)))  # 边权重

                    f.write(" ".join(line_parts) + "\n")

            # ==========================================
            # 分支 B: 超图 (HGPA/MCLA 用这个, method=2/3)
            # ==========================================
            else:
                # 移除空列 (没有节点的超边)
                col_sums = np.sum(e, axis=0)
                valid_columns = np.where(col_sums > 0)[0]

                if len(valid_columns) != e.shape[1]:
                    e = e[:, valid_columns]
                    if method == 3 and w is not None and len(w) == len(col_sums):
                        w = np.array(w)[valid_columns]

                # Header: NumberOfHyperEdges NumberOfVertices Format(1=EdgeWeights)
                f.write(f"{e.shape[1]} {e.shape[0]} 1\n")

                # 遍历每个超边 (列)
                for i in range(e.shape[1]):
                    # 找到该超边包含的节点 (行)
                    edges_indices = np.where(e[:, i] > 0)[0]

                    if method == 2:
                        weight = np.sum(e[:, i])
                    else:  # method == 3
                        weight = w[i] if w is not None and len(w) > i else 1

                    line_parts = [str(int(weight))]

                    # 写入节点索引 (1-based)
                    for idx in edges_indices:
                        line_parts.append(str(idx + 1))

                    f.write(" ".join(line_parts) + "\n")

        return dataname

    except IOError as e:
        print(f"wgraph: writing to {dataname} failed: {e}")
        return None

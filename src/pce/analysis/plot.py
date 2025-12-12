import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Dict, Optional


def plot(
        data: Union[str, List[Dict], pd.DataFrame],
        metric: str,
        title: Optional[str] = None,
        xlabel: str = "Round",
        ylabel: Optional[str] = None,
        output_dir: str = "./plots",
        file_name: Optional[str] = None
):
    """
    绘制指定指标随实验轮次变化的趋势图。

    Args:
        data: 数据源。可以是 CSV 文件路径、结果列表 (List[Dict]) 或 DataFrame。
        metric: 要绘制的指标名称 (例如 'ACC', 'NMI')。只接收一个字符串参数。
        title: 图表标题。如果为 None，默认生成 "Trend of {metric}"。
        xlabel: X 轴标签，默认为 "Round"。
        ylabel: Y 轴标签。如果为 None，默认使用 metric 名称。
        output_dir: 图片保存目录。
        file_name: 图片保存文件名。如果为 None，默认生成 "{metric}_trend.png"。
    """

    # --- 1. 数据加载与标准化 ---
    try:
        if isinstance(data, str):
            # 如果是 CSV 文件路径
            if os.path.exists(data):
                df = pd.read_csv(data)
            else:
                raise FileNotFoundError(f"File not found: {data}")
        elif isinstance(data, list):
            # 如果是内存中的结果列表 (res)
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("Data must be a csv path, list of dicts, or DataFrame.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- 2. 检查指标是否存在 ---
    if metric not in df.columns:
        print(f"Error: Metric '{metric}' not found in data. Available columns: {list(df.columns)}")
        return

    # --- 3. 准备绘图数据 ---
    # 构造 X 轴数据：如果有 'Round' 列就用，没有就用索引 (1-based)
    if 'Round' in df.columns:
        x_data = df['Round']
    else:
        x_data = range(1, len(df) + 1)

    y_data = df[metric]

    # --- 4. 绘图配置 ---
    # 设置风格
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))

    # 绘制折线图
    sns.lineplot(x=x_data, y=y_data, marker='o', linewidth=2, label=metric)

    # 设置标签和标题
    final_title = title if title else f"Trend of {metric}"
    final_ylabel = ylabel if ylabel else metric
    final_filename = file_name if file_name else f"{metric}_trend.png"

    plt.title(final_title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(final_ylabel, fontsize=12)

    # 强制 X 轴显示整数刻度 (因为是轮次)
    from matplotlib.ticker import MaxNLocator
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend()
    plt.tight_layout()

    # --- 5. 保存图片 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    save_path = os.path.join(output_dir, final_filename)

    try:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")
    except Exception as e:
        print(f"Failed to save plot: {e}")
    finally:
        plt.close()  # 关闭画布，释放内存
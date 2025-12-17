import os

import matplotlib.pyplot as plt
import seaborn as sns


def set_paper_style(context='paper', style='ticks', font_scale=1.2):
    """
    设置符合学术发表要求的绘图风格 (Times New Roman, 去除多余边框等)。

    Args:
        context (str): 'paper', 'notebook', 'talk', 'poster' (seaborn context)
        style (str): 'white', 'dark', 'whitegrid', 'darkgrid', 'ticks'
        font_scale (float): 字体缩放比例
    """
    # 1. 设置 Seaborn 基础风格
    sns.set_context(context, font_scale=font_scale)
    sns.set_style(style)

    # 2. 强制设置字体为 Times New Roman 或类似衬线字体
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",  # 数学公式字体风格
        "axes.grid": True,  # 默认开启网格
        "grid.linestyle": "--",  # 网格虚线
        "grid.alpha": 0.6,  # 网格透明度
        "xtick.direction": "in",  # 刻度向内
        "ytick.direction": "in"
    })


def save_fig(fig, path, dpi=300):
    """
    统一保存逻辑，自动去除白边，修复 PDF 损坏问题
    """
    if not path:
        return

    # 1. 关键修复：设置 PDF 字体类型为 42 (TrueType)
    # 这能解决 99% 的 PDF 打不开或无法编辑的问题
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # 2. 自动确保保存路径的文件夹存在 (防止 FileNotFoundError)
    dirname = os.path.dirname(os.path.abspath(path))
    if dirname and not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError:
            pass  # 忽略并发创建时的错误

    # 3. 智能获取文件后缀 (如 .pdf, .png)
    _, ext = os.path.splitext(path)
    fmt = ext.lower().strip('.')

    # 如果没有后缀，默认给个 png
    if not fmt:
        fmt = 'png'
        path = f"{path}.png"

    try:
        # 4. 显式传入 format 参数
        fig.savefig(path, bbox_inches='tight', dpi=dpi, format=fmt)
        # print(f"[Save] Figure saved to: {path}")
    except Exception as e:
        print(f"[Error] Failed to save figure: {e}")

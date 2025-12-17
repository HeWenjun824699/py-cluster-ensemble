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
    统一保存逻辑，自动去除白边
    """
    if path:
        fig.savefig(path, bbox_inches='tight', dpi=dpi)
        # print(f"[Analysis] Figure saved to {path}")

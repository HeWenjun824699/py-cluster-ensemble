import os

import matplotlib.pyplot as plt
import seaborn as sns


def set_paper_style(context='paper', style='ticks', font_scale=1.2):
    """
    Set plot style meeting academic publication requirements (Times New Roman, remove excess borders, etc.).

    Args:
        context (str): 'paper', 'notebook', 'talk', 'poster' (seaborn context)
        style (str): 'white', 'dark', 'whitegrid', 'darkgrid', 'ticks'
        font_scale (float): Font scaling factor
    """
    # 1. Set Seaborn base style
    sns.set_context(context, font_scale=font_scale)
    sns.set_style(style)

    # 2. Force font to Times New Roman or similar serif font
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",  # Math formula font style
        "axes.grid": True,  # Enable grid by default
        "grid.linestyle": "--",  # Grid line style
        "grid.alpha": 0.6,  # Grid transparency
        "xtick.direction": "in",  # Ticks inward
        "ytick.direction": "in"
    })


def save_fig(fig, path, dpi=300):
    """
    Unified save logic, automatically remove white borders, fix PDF corruption issues
    """
    if not path:
        return

    # 1. Key fix: Set PDF font type to 42 (TrueType)
    # This solves 99% of issues where PDFs cannot be opened or edited
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # 2. Automatically ensure the save path folder exists (prevent FileNotFoundError)
    dirname = os.path.dirname(os.path.abspath(path))
    if dirname and not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError:
            pass  # Ignore errors during concurrent creation

    # 3. Smartly get file extension (e.g., .pdf, .png)
    _, ext = os.path.splitext(path)
    fmt = ext.lower().strip('.')

    # If no extension, default to png
    if not fmt:
        fmt = 'png'
        path = f"{path}.png"

    try:
        # 4. Explicitly pass format parameter
        fig.savefig(path, bbox_inches='tight', dpi=dpi, format=fmt)
        # print(f"[Save] Figure saved to: {path}")
    except Exception as e:
        print(f"[Error] Failed to save figure: {e}")

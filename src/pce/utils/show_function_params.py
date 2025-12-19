import inspect
from .. import io, generators, consensus, metrics, pipelines, grid, analysis

# 1.定义不可搜索参数（黑名单）
NON_SEARCHABLE_PARAMS = {
    'BPs', 'Y', 'X', 'labels', 'verbose', 'n_jobs', 'debug'
    'file_path', 'input_data', 'input_dir', 'output_dir'
}

# 2.定义固定参数
FIXED_PARAMS = {
    'nClusters', 'nPartitions', 'seed', 'maxiter', 'replicates', 'nBase', 'nRepeat',
    'save_format', 'consensus_method', 'generator_method',
    'overwrite', 'module_type', 'default_name'
}

# 3.建立模块映射表（方便扩展）
MODULE_MAP = {
    'io': io,
    'generators': generators,
    'consensus': consensus,
    'metrics': metrics,
    'pipelines': pipelines,
    'grid': grid,
    'analysis': analysis
}


def _get_visual_width(s: str) -> int:
    """计算字符串的视觉宽度 (Emoji 算 2 格)"""
    width = len(s)
    # 如果包含宽字符 Emoji，手动增加宽度计数
    if '🔒' in s: width += 1
    if '✅' in s: width += 1
    return width


def show_function_params(method_name: str, module_type: str = 'consensus'):
    """
    智能打印算法参数 (已修复 Emoji 对齐问题)
    """
    target_module = MODULE_MAP.get(module_type)

    try:
        func = getattr(target_module, method_name)
    except AttributeError:
        print(f"[Error] Method '{method_name}' not found in pce.{module_type}")
        return

    sig = inspect.signature(func)

    print(f"\n[Info] Parameter Status for '{method_name}' ({module_type}):")

    # --- 1. 定义固定列宽 ---
    W_NAME = 20
    W_ROLE = 26  # 给 Role 列留足够的空间
    W_VAL = 18
    W_TYPE = 15

    # --- 2. 打印表头 ---
    header = f"{'Parameter':<{W_NAME}} | {'Role':<{W_ROLE}} | {'Default Value':<{W_VAL}} | {'Type Hint'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for name, param in sig.parameters.items():

        # --- A. 获取 Type Hint ---
        annotation = param.annotation
        if annotation is not inspect.Parameter.empty:
            type_str = getattr(annotation, '__name__', str(annotation))
        elif param.default is not inspect.Parameter.empty:
            type_str = type(param.default).__name__ + " (inferred)"
        else:
            type_str = "Any"

        # --- B. 判断 Role 和 Value ---
        # 情况 A: 🔒 Fixed
        if name in FIXED_PARAMS:
            role = "🔒 Fixed Parameter"
            if param.default is not inspect.Parameter.empty:
                current_val = f"{param.default}"
            else:
                current_val = "(No default)"

        # 情况 B: [Input / Output]
        elif name in NON_SEARCHABLE_PARAMS:
            if name in ['verbose', 'n_jobs', 'debug']:
                role = "[System Control]"
            elif name == 'input_dir':
                role = "[Input Dir]"
            elif name == 'output_dir':
                role = "[Output Dir]"
            elif name == 'file_path':
                role = "[File Path]"
            else:
                role = "[Input Data]"
            current_val = "(Required)"

        # 情况 C: (Required)
        elif param.default == inspect.Parameter.empty:
            if param.kind in [inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]:
                role = "[Optional Args]"
                current_val = "(Optional)"
            else:
                role = "[Input / Required]"
                current_val = "(Required)"

        # 情况 D: ✅ Searchable
        else:
            role = "✅ Hyperparameter"
            current_val = f"{param.default} (Default)"

        # --- C. 格式化输出 (核心修复) ---

        # 1. 截断过长的 Value
        if len(current_val) > W_VAL - 2:
            current_val = current_val[:W_VAL - 4] + "..."

        # 2. 处理 **kwargs
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            name = f"**{name}"
            type_str = "Dict"

        # 3. [手动计算对齐]
        # 计算 role 的视觉宽度
        visual_w = _get_visual_width(role)
        # 计算需要填充多少个空格
        padding = max(0, W_ROLE - visual_w)
        # 手动拼接: 内容 + 空格
        role_str = role + " " * padding

        # 4. 打印
        # 注意: role_str 已经包含了填充空格，所以这里不需要再写 :<26，直接放进去即可
        print(f"{name:<{W_NAME}} | {role_str} | {current_val:<{W_VAL}} | {type_str}")

    print("-" * len(header))
    print("Legend: 🔒 = Fixed in your config | ✅ = Available for Grid Search")

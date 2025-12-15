import inspect
from .. import consensus, generators


def show_algorithm_params(method_name: str, module_type: str = 'consensus'):
    """
    打印指定算法支持的参数（优化版：显示真实数据类型）
    """
    target_module = consensus if module_type == 'consensus' else generators

    try:
        func = getattr(target_module, method_name)
    except AttributeError:
        print(f"[Error] Method '{method_name}' not found in pce.{module_type}")
        return

    sig = inspect.signature(func)

    print(f"\n[Info] Parameters for '{method_name}' ({module_type}):")
    # 调整表头：增加 Type Hint 列
    print(f"{'Parameter':<20} | {'Default':<15} | {'Type Hint'}")
    print("-" * 60)

    for name, param in sig.parameters.items():
        # if name in ['BPs', 'Y', 'X']: continue

        # 1. 获取默认值
        default_val = param.default
        if default_val == inspect.Parameter.empty:
            def_str = "(Required)"
        else:
            def_str = str(default_val)

        # 2. 获取数据类型 (核心修改)
        # 优先读取函数定义的 Type Hint (例如: seed: int)
        annotation = param.annotation

        if annotation is not inspect.Parameter.empty:
            # 获取类名 (如 'int', 'str', 'List')
            type_str = getattr(annotation, '__name__', str(annotation))
        elif default_val is not inspect.Parameter.empty:
            # 如果没有 Type Hint，但有默认值，显示默认值的类型 (例如: 20 -> int)
            type_str = type(default_val).__name__ + " (inferred)"
        else:
            type_str = "Any"

        # 3. 处理可变参数显示的优化 (**kwargs)
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            name = f"**{name}"
            type_str = "Dict"
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            name = f"*{name}"
            type_str = "Tuple"

        print(f"{name:<20} | {def_str:<15} | {type_str}")

import inspect
from .. import io, generators, consensus, metrics, pipelines, grid, analysis

# 1. Define non-searchable parameters (blacklist)
NON_SEARCHABLE_PARAMS = {
    'BPs', 'Y', 'X', 'labels', 'verbose', 'n_jobs', 'debug'
    'file_path', 'input_data', 'input_dir', 'output_dir'
}

# 2. Define fixed parameters
FIXED_PARAMS = {
    'nClusters', 'nPartitions', 'seed', 'maxiter', 'replicates', 'nBase', 'nRepeat',
    'save_format', 'consensus_method', 'generator_method',
    'overwrite', 'module_type', 'default_name'
}

# 3. Build module map (for extensibility)
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
    """Calculate visual width of string (Emoji counts as 2)"""
    width = len(s)
    # If wide character Emoji is included, manually increase width count
    if '🔒' in s: width += 1
    if '✅' in s: width += 1
    return width


def show_function_params(method_name: str, module_type: str = 'consensus'):
    """
    Smartly print algorithm parameters (Emoji alignment fixed)
    """
    target_module = MODULE_MAP.get(module_type)

    try:
        func = getattr(target_module, method_name)
    except AttributeError:
        print(f"[Error] Method '{method_name}' not found in pce.{module_type}")
        return

    sig = inspect.signature(func)

    print(f"\n[Info] Parameter Status for '{method_name}' ({module_type}):")

    # --- 1. Define fixed column widths ---
    W_NAME = 20
    W_ROLE = 26  # Leave enough space for Role column
    W_VAL = 18
    W_TYPE = 15

    # --- 2. Print header ---
    header = f"{'Parameter':<{W_NAME}} | {'Role':<{W_ROLE}} | {'Default Value':<{W_VAL}} | {'Type Hint'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for name, param in sig.parameters.items():

        # --- A. Get Type Hint ---
        annotation = param.annotation
        if annotation is not inspect.Parameter.empty:
            type_str = getattr(annotation, '__name__', str(annotation))
        elif param.default is not inspect.Parameter.empty:
            type_str = type(param.default).__name__ + " (inferred)"
        else:
            type_str = "Any"

        # --- B. Determine Role and Value ---
        # Case A: 🔒 Fixed
        if name in FIXED_PARAMS:
            role = "🔒 Fixed Parameter"
            if param.default is not inspect.Parameter.empty:
                current_val = f"{param.default}"
            else:
                current_val = "(No default)"

        # Case B: [Input / Output]
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

        # Case C: (Required)
        elif param.default == inspect.Parameter.empty:
            if param.kind in [inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]:
                role = "[Optional Args]"
                current_val = "(Optional)"
            else:
                role = "[Input / Required]"
                current_val = "(Required)"

        # Case D: ✅ Searchable
        else:
            role = "✅ Hyperparameter"
            current_val = f"{param.default} (Default)"

        # --- C. Format output (core fix) ---

        # 1. Truncate overly long Value
        if len(current_val) > W_VAL - 2:
            current_val = current_val[:W_VAL - 4] + "..."

        # 2. Handle **kwargs
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            name = f"**{name}"
            type_str = "Dict"

        # 3. [Manual alignment calculation]
        # Calculate visual width of role
        visual_w = _get_visual_width(role)
        # Calculate padding spaces needed
        padding = max(0, W_ROLE - visual_w)
        # Manual concatenation: content + spaces
        role_str = role + " " * padding

        # 4. Print
        # Note: role_str already includes padding, so no need for :<26 here
        print(f"{name:<{W_NAME}} | {role_str} | {current_val:<{W_VAL}} | {type_str}")

    print("-" * len(header))
    print("Legend: 🔒 = Fixed in your config | ✅ = Available for Grid Search")

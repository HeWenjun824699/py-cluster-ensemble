import pce


# 1. Check available parameters for a specific method
pce.utils.show_function_params('lwea', module_type='consensus')

# 2. Define paths
data_dir = r"./data/CDKM200"
out_dir = r"./results/grid/GridSearch_001"

# 3. Grid Parameters
param_grid = {
    'consensus_method': 'lwea',
    'theta': [1, 5, 10, 20],
    'lamb': [10, 50, 100, 200],
}

# 4. Fixed Parameters (Constant configuration)
fixed_params = {
    'generator_method': 'cdkmeans',
    'nPartitions': 200,     # generator param
    'seed': 2026,           # shared param
    'maxiter': 100,         # generator param
    'replicates': 1,        # generator param
    'nBase': 20,            # consensus param
    'nRepeat': 10           # consensus param
}

# 5. Execute Grid Search
searcher = pce.grid.GridSearcher(data_dir, out_dir)
searcher.run(param_grid, fixed_params)

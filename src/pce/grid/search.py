import json
import time
import inspect
import logging
import itertools
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple

# 引入核心组件
from .. import io
from .. import consensus
from .. import generators
from .. import metrics


class GridSearcher:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _filter_kwargs(self, func, all_params: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        核心魔法：根据函数的签名，从大字典中提取它需要的参数。
        返回: (valid_params, ignored_keys)
        """
        sig = inspect.signature(func)
        valid_params = {}
        ignored_keys = []

        func_params = sig.parameters.keys()

        for k, v in all_params.items():
            if k in func_params:
                valid_params[k] = v
            elif k not in ['consensus_method', 'generator_method']:
                # 不记录 method 本身，只记录真正的参数
                ignored_keys.append(k)

        return valid_params, ignored_keys

    def _setup_experiment_logger(self, log_path: Path) -> logging.Logger:
        """为每个实验文件夹创建一个独立的 Logger"""
        logger = logging.getLogger(str(log_path))
        logger.setLevel(logging.INFO)
        logger.handlers = []  # 清除旧句柄

        # File Handler (写入文件夹内的 run.log)
        fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

        return logger

    def run(self, param_grid: Dict[str, List[Any]], fixed_params: Dict[str, Any] = None):
        if fixed_params is None: fixed_params = {}

        # 1. 生成参数组合
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        print(f">>> Start Grid Search. Total combinations: {len(combinations)}")
        print(f">>> Output Directory: {self.output_dir}\n")

        all_summary = []  # 汇总表数据

        # 2. 遍历数据集
        mat_files = list(self.input_dir.glob("*.mat"))

        for file_idx, file_path in enumerate(mat_files):
            dataset_name = file_path.stem
            print(f"[{file_idx + 1}/{len(mat_files)}] Dataset: {dataset_name}")

            # 2.1 数据集根目录
            ds_output_dir = self.output_dir / dataset_name
            ds_output_dir.mkdir(exist_ok=True)

            # 2.2 加载数据 (Cache)
            try:
                # 尝试加载 BPs
                try:
                    BPs_cached, Y_cached = io.load_mat_BPs_Y(file_path)
                    data_source = "precomputed"
                except IOError:
                    X_cached, Y_cached = io.load_mat_X_Y(file_path)
                    data_source = "raw"
                    BPs_cached = None
            except Exception as e:
                print(f"  [Error] Failed to load {dataset_name}: {e}")
                continue

            # 3. 遍历参数组合
            for combo_idx, params in enumerate(combinations):
                # 3.1 准备配置
                # 命名规则: Exp_{ID}_{Method}
                c_method = params.get('consensus_method', fixed_params.get('consensus_method', 'unknown'))
                exp_id = f"Exp_{combo_idx + 1:03d}_{c_method}"
                exp_dir = ds_output_dir / exp_id
                exp_dir.mkdir(exist_ok=True)

                # 合并所有参数
                full_config = {**fixed_params, **params}

                # 3.2 初始化单次实验日志
                logger = self._setup_experiment_logger(exp_dir / "run.log")
                logger.info(f"=== Experiment: {exp_id} ===")
                logger.info(f"Dataset: {dataset_name}")
                logger.info(f"Data Source: {data_source}")
                logger.info(f"Full Config: {full_config}")

                start_time = time.time()
                status = "SUCCESS"
                error_msg = ""
                metrics_res = {}

                try:
                    BPs = None
                    Y = None

                    # --- A. 准备/生成基聚类 (Generators) ---
                    g_method = full_config.get('generator_method', 'cdkmeans')

                    if data_source == "precomputed":
                        BPs, Y = BPs_cached, Y_cached
                        logger.info("Using precomputed BPs from .mat file.")
                    else:
                        logger.info(f"Generating BPs using {g_method}...")
                        gen_func = getattr(generators, g_method)

                        # 【参数自动过滤】
                        gen_kwargs, gen_ignored = self._filter_kwargs(gen_func, full_config)
                        logger.info(f"Generator params used: {gen_kwargs}")
                        if gen_ignored: logger.info(f"Generator ignored params: {gen_ignored}")

                        BPs, Y = gen_func(X_cached, Y_cached, **gen_kwargs)

                    # --- B. 聚类集成 (Consensus) ---
                    logger.info(f"Running Consensus: {c_method}...")
                    con_func = getattr(consensus, c_method)

                    # 【参数自动过滤】
                    con_kwargs, con_ignored = self._filter_kwargs(con_func, full_config)
                    logger.info(f"Consensus params used: {con_kwargs}")
                    if con_ignored: logger.info(f"Consensus ignored params: {con_ignored}")

                    labels, _ = con_func(BPs, Y, **con_kwargs)

                    # --- C. 评估与保存 ---
                    metrics_res = metrics.evaluation_batch(labels, Y)
                    logger.info(f"Metrics: {metrics_res}")

                    # 保存结果文件
                    # 1. Labels
                    pd.DataFrame(labels).to_csv(exp_dir / "labels.csv", index=False, header=False)
                    # 2. Config
                    with open(exp_dir / "config.json", 'w') as f:
                        json.dump(full_config, f, indent=4)
                    # 3. Scores
                    with open(exp_dir / "scores.json", 'w') as f:
                        json.dump(metrics_res, f, indent=4)

                except Exception as e:
                    status = "FAILED"
                    error_msg = str(e)
                    logger.error(f"Experiment failed: {e}", exc_info=True)
                    print(f"  x {exp_id} Failed. See log.")

                # 释放 logger 句柄
                handlers = logger.handlers[:]
                for handler in handlers:
                    handler.close()
                    logger.removeHandler(handler)

                # --- 4. 记录到总表 ---
                elapsed = time.time() - start_time
                summary_record = {
                    "dataset": dataset_name,
                    "exp_id": exp_id,
                    "status": status,
                    "elapsed": round(elapsed, 4),
                    **params,  # 仅记录变动参数，避免表格太宽
                    **metrics_res  # 结果
                }
                all_summary.append(summary_record)

                if status == "SUCCESS":
                    nmi = metrics_res.get('NMI', 0)
                    print(f"  - {exp_id}: NMI={nmi:.4f} ({elapsed:.2f}s)")

        # 5. 保存总汇总表
        if all_summary:
            df = pd.DataFrame(all_summary)
            df.to_csv(self.output_dir / "grid_summary.csv", index=False)
            print(f"\nAll Done. Summary saved to {self.output_dir / 'grid_summary.csv'}")


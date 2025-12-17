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
from .. import analysis


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
        if func is None:
            return {}, []

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

    def _compute_avg_metrics(self, metrics_res: Any) -> Dict[str, float]:
        """内部函数：处理 metrics 结果。"""
        if isinstance(metrics_res, list):
            if not metrics_res:
                return {}
            df = pd.DataFrame(metrics_res)
            return df.mean().to_dict()
        elif isinstance(metrics_res, dict):
            return metrics_res
        return {}

    def _prune_combinations(self, raw_combinations: List[Dict], fixed_params: Dict) -> List[Dict]:
        """
        【新增功能】智能去重 & 参数清洗
        逻辑：
        1. 去重：如果参数的变化对于当前的 generator 和 consensus 都是无效的，则跳过。
        2. 清洗：在保留的任务中，剔除掉当前方法不需要的冗余参数。
        """
        valid_tasks = []
        seen_signatures = set()

        print(f"\n>>> Pruning tasks... (Raw combinations: {len(raw_combinations)})")

        for params in raw_combinations:
            full_config = {**fixed_params, **params}

            # 1. 获取当前的方法名
            c_method_name = full_config.get('consensus_method', 'unknown')
            g_method_name = full_config.get('generator_method', 'cdkmeans')

            # 2. 获取函数对象
            c_func = getattr(consensus, c_method_name, None)
            g_func = getattr(generators, g_method_name, None)

            # 3. 提取有效参数 (Effective Params)
            c_valid, _ = self._filter_kwargs(c_func, full_config)
            g_valid, _ = self._filter_kwargs(g_func, full_config)

            # 4. 生成唯一指纹 (Signature)
            # 指纹由：Generator名 + Generator有效参数 + Consensus名 + Consensus有效参数 组成
            # 如果两个组合生成的指纹一样，说明它们跑出来的结果绝对是一样的（数学上等价）
            signature = (
                g_method_name,
                json.dumps(g_valid, sort_keys=True),
                c_method_name,
                json.dumps(c_valid, sort_keys=True)
            )

            if signature not in seen_signatures:
                seen_signatures.add(signature)

                # ==========================================
                # NEW: 参数清洗逻辑
                # ==========================================
                # 有效的键 = (Consensus需要的参数) + (Generator需要的参数) + (方法名本身)
                allowed_keys = set(c_valid.keys()) | set(g_valid.keys()) | {'consensus_method', 'generator_method'}

                # 只保留 params 中存在的、且在 allowed_keys 中的参数
                cleaned_params = {k: v for k, v in params.items() if k in allowed_keys}

                valid_tasks.append(cleaned_params)
                # ==========================================
            else:
                # # 调试用：可以看到哪些被跳过了
                # 1. 找出被忽略的参数名 (集合运算)
                ignored_keys = set(params.keys()) - set(c_valid.keys()) - set(g_valid.keys())

                # 2. 提取这些参数对应的值 (字典推导式)
                ignored_details = {k: params[k] for k in ignored_keys}

                # 3. 打印详细信息
                if ignored_details:
                    print(f"    [Skipped] Redundant config for '{c_method_name}': {ignored_details}")

        print(f">>> Pruning finished. Effective tasks: {len(valid_tasks)} (Removed {len(raw_combinations) - len(valid_tasks)} redundant tasks)")
        return valid_tasks

    def run(self, param_grid: Dict[str, List[Any]], fixed_params: Dict[str, Any] = None):
        if fixed_params is None: fixed_params = {}

        # 1. 生成原始参数组合
        keys = param_grid.keys()
        values = param_grid.values()
        raw_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # 【Step 1.5: 智能去重】
        # 这里进行去重，只保留真正有意义的组合
        tasks = self._prune_combinations(raw_combinations, fixed_params)

        print(f"\n>>> Start Grid Search.")
        print(f">>> Output Directory: {self.output_dir}")

        all_summary = []  # 汇总表数据

        # 2. 遍历数据集
        mat_files = list(self.input_dir.glob("*.mat"))

        for file_idx, file_path in enumerate(mat_files):
            dataset_name = file_path.stem
            print(f"\n[{file_idx + 1}/{len(mat_files)}] Dataset: {dataset_name}")

            # 2.1 数据集根目录
            ds_output_dir = self.output_dir / dataset_name
            ds_output_dir.mkdir(exist_ok=True)

            # 2.2 加载数据 (Cache)
            try:
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

            # 3. 遍历去重后的任务列表 (Tasks)
            for task_idx, params in enumerate(tasks):
                # 3.1 准备配置
                # 命名建议：不再单纯用 combo_idx，而是最好体现这组参数的特征，或者直接用 task_idx
                c_method = params.get('consensus_method', fixed_params.get('consensus_method', 'unknown'))
                # 使用 task_idx + 1 保证顺序编号
                exp_id = f"Exp_{task_idx + 1:03d}_{c_method}"
                exp_dir = ds_output_dir / exp_id
                exp_dir.mkdir(exist_ok=True)

                # 合并所有参数
                full_config = {**fixed_params, **params}

                # 3.2 初始化单次实验日志
                logger = self._setup_experiment_logger(exp_dir / "run.log")
                logger.info(f"===================== Experiment: {exp_id} =====================")
                logger.info(f"Dataset: {dataset_name}")
                logger.info(f"Params: {full_config}")

                start_time = time.time()
                status = "SUCCESS"
                metrics_avg = {}

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
                        gen_kwargs, _ = self._filter_kwargs(gen_func, full_config)
                        BPs, Y = gen_func(X_cached, Y_cached, **gen_kwargs)

                    # --- B. 聚类集成 (Consensus) ---
                    logger.info(f"Running Consensus: {c_method}...")
                    con_func = getattr(consensus, c_method)
                    con_kwargs, _ = self._filter_kwargs(con_func, full_config)
                    labels, _ = con_func(BPs, Y, **con_kwargs)
                    logger.info(f"Labels: {labels}")

                    # --- C. 评估与保存 ---
                    metrics_res = metrics.evaluation_batch(labels, Y)
                    logger.info(f"Metrics (Raw): {metrics_res}")
                    # 计算各个指标的平均值
                    metrics_avg = self._compute_avg_metrics(metrics_res)
                    logger.info(f"Metrics (Avg): {metrics_avg}")

                    # 绘制指标折线图
                    plot_metrics = ["ACC", "NMI", "Purity", "AR", "RI", "MI", "HI", "F-Score", "Precision", "Recall", "Entropy", "SDCS", "RME", "Bal"]
                    analysis.plot_metric_line(results_list=metrics_res, metrics=plot_metrics, title=f"{dataset_name} - {exp_id}", save_path=f"{exp_dir / 'line_plot.png'}")
                    logger.info(f"Line plot saved in {exp_dir / 'line_plot.png'}")

                    # 保存参数
                    with open(exp_dir / "params.json", 'w') as f:
                        json.dump(full_config, f, indent=4)
                    logger.info(f"Params saved in {exp_dir / 'params.json'}")
                    # 保存标签结果
                    pd.DataFrame(labels).T.to_csv(exp_dir / "labels.csv", index=False, header=False)
                    logger.info(f"Labels saved in {exp_dir / 'labels.csv'}")
                    # 保存指标
                    with open(exp_dir / "metrics.json", 'w') as f:
                        json.dump(metrics_res, f, indent=4)
                    logger.info(f"Scores saved in {exp_dir / 'metrics.json'}")
                    # 保存结果
                    io.save_results_csv(data=metrics_res, output_path=f"{exp_dir / 'results.csv'}")
                    logger.info(f"Results saved in {exp_dir / 'results.csv'}")

                    logger.info(f"===================== Experiment {exp_id} Completed. =====================")

                except Exception as e:
                    status = "FAILED"
                    logger.error(f"Experiment failed: {e}", exc_info=True)
                    print(f"  x {exp_id} Failed. See log.")

                # 清理 Logger
                handlers = logger.handlers[:]
                for handler in handlers:
                    handler.close()
                    logger.removeHandler(handler)

                # --- 4. 记录到总表 ---
                elapsed = time.time() - start_time
                summary_record = {
                    "Dataset": dataset_name,
                    "Exp_id": exp_id,
                    "Status": status,
                    "Time": round(elapsed, 4),
                    **params,
                    **metrics_avg
                }
                all_summary.append(summary_record)

                if status == "SUCCESS":
                    acc = metrics_avg.get('ACC', 0)
                    nmi = metrics_avg.get('NMI', 0)
                    ari = metrics_avg.get('AR', 0)
                    # 简单打印进度
                    print(f"  - {exp_id}: ACC={acc:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}  Time={elapsed:.4f}s")

        # 5. 保存总汇总表
        if all_summary:
            df = pd.DataFrame(all_summary)
            df.to_csv(self.output_dir / "grid_summary.csv", index=False)
            print(f"\nAll Done. Summary saved to {self.output_dir / 'grid_summary.csv'}")

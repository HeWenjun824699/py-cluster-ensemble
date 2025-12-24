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
            # c_method_name, g_method_name 转小写
            c_method_name = c_method_name.lower()
            g_method_name = g_method_name.lower()

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

            # =======================================================
            # 1. 预先计算当前网格中属于各方法的“变动参数”
            #    (放在 if/else 之前，确保两个分支都能用)
            # =======================================================
            g_grid_params = {k: v for k, v in params.items() if k in g_valid}
            c_grid_params = {k: v for k, v in params.items() if k in c_valid}

            # 2. 格式化显示的字符串（定义一个小函数或直接写逻辑）
            # 如果字典有值显示 "name{...}"，没值只显示 "name"
            g_str = f"{g_method_name}{g_grid_params}" if g_grid_params else g_method_name
            c_str = f"{c_method_name}{c_grid_params}" if c_grid_params else c_method_name

            if signature not in seen_signatures:
                seen_signatures.add(signature)

                # --- 2. 参数清洗 (存入任务列表) ---
                allowed_keys = set(c_valid.keys()) | set(g_valid.keys()) | {'consensus_method', 'generator_method'}
                cleaned_params = {k: v for k, v in params.items() if k in allowed_keys}
                valid_tasks.append(cleaned_params)

                # --- 3. [主日志] 新任务 ---
                # 使用 g_grid_params 仅展示当前网格搜索中变化的参数，简洁明了
                print(f"  + [Add Task {len(valid_tasks)}] Config: {g_str} + {c_str}")

            else:
                # --- 4. [次日志] 重复任务 ---
                # 计算被忽略的冗余参数
                ignored_keys = set(params.keys()) - set(c_valid.keys()) - set(g_valid.keys())
                ignored_details = {k: params[k] for k in ignored_keys}

                if ignored_details:
                    # 同样使用 g_grid_params 展示归属，清晰指出是哪个组合的重复项
                    print(f"    - [Skipped] Duplicate of {g_str} + {c_str}. Ignored: {ignored_details}")
                else:
                    print(f"    - [Skipped] Exact duplicate input.")

        print(f">>> Pruning finished. Effective tasks: {len(valid_tasks)} (Removed {len(raw_combinations) - len(valid_tasks)} redundant tasks)")
        return valid_tasks

    def run(self, param_grid: Dict[str, List[Any]], fixed_params: Dict[str, Any] = None):
        if fixed_params is None: fixed_params = {}

        # ================= NEW: 强制固定 consensus_method =================
        # 逻辑：无论用户是否在 param_grid 里写了 consensus_method，
        # 都将其移动到 fixed_params，防止发生笛卡尔积或字符串拆解。
        if 'consensus_method' in param_grid:
            val = param_grid.pop('consensus_method')

            # 情况 A: 用户传入的是列表 (e.g., ["CSPA"]) -> 取第一个元素
            if isinstance(val, list) and len(val) > 0:
                fixed_params['consensus_method'] = val[0]
                # print(f"\n[Info] 'consensus_method' ({val[0]}) moved from grid to fixed_params.")
            # 情况 B: 用户传入的是纯字符串 (e.g., "CSPA") -> 直接使用
            elif isinstance(val, str):
                fixed_params['consensus_method'] = val
                # print(f"\n[Info] 'consensus_method' ({val}) moved from grid to fixed_params.")
        # =================================================================

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

            current_dataset_summary = []  # 当前数据集的汇总表数据

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

            # =======================================================
            # 【修改点 1】: 初始化运行时缓存变量
            # =======================================================
            last_gen_sig = None  # 上一次生成器的“指纹”
            current_BPs_cache = None  # 当前内存中的 BPs 数据
            # =======================================================

            # 3. 遍历去重后的任务列表 (Tasks)
            for task_idx, params in enumerate(tasks):
                # 3.1 准备配置
                # 命名建议：不再单纯用 combo_idx，而是最好体现这组参数的特征，或者直接用 task_idx
                c_method = params.get('consensus_method', fixed_params.get('consensus_method', 'unknown'))
                # 转小写
                c_method = c_method.lower()

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

                    # --- A. 准备/生成基聚类 (带缓存逻辑) ---
                    if data_source == "precomputed":
                        BPs, Y = BPs_cached, Y_cached
                        logger.info("Using precomputed BPs from .mat file.")
                    else:
                        # 1. 提取当前任务所需的生成器参数
                        g_method = full_config.get('generator_method', 'cdkmeans')
                        # 转小写
                        g_method = g_method.lower()
                        g_func = getattr(generators, g_method)
                        g_kwargs, _ = self._filter_kwargs(g_func, full_config)
                        # 2. 生成当前任务的“指纹” (方法名 + 参数JSON)
                        # sort_keys=True 保证字典顺序不同但内容相同时指纹一致
                        current_sig = (g_method, json.dumps(g_kwargs, sort_keys=True))
                        # 3. 检查缓存
                        if current_sig == last_gen_sig and current_BPs_cache is not None:
                            # 【命中缓存】
                            logger.info(">> [Cache Hit] Reusing BPs from previous task (Generator params unchanged).")
                            BPs = current_BPs_cache
                        else:
                            # 【未命中，重新生成】
                            logger.info(f">> [Cache Miss] Generating BPs using {g_method}...")
                            gen_start = time.time()
                            BPs = g_func(X_cached, Y_cached, **g_kwargs)
                            gen_time = time.time() - gen_start
                            logger.info(f"   Generation completed in {gen_time:.4f}s")

                            # 更新缓存
                            current_BPs_cache = BPs
                            last_gen_sig = current_sig
                        Y = Y_cached

                    # --- B. 聚类集成 (Consensus) ---
                    logger.info(f"Running Consensus: {c_method}...")
                    con_func = getattr(consensus, c_method)
                    con_kwargs, _ = self._filter_kwargs(con_func, full_config)
                    labels, time_list = con_func(BPs, Y, **con_kwargs)
                    logger.info(f"Labels: {labels}")

                    # --- C. 评估与保存 ---
                    metrics_res = metrics.evaluation_batch(labels, Y, time_list)
                    logger.info(f"Metrics (Raw): {metrics_res}")
                    # 计算各个指标的平均值
                    metrics_avg = self._compute_avg_metrics(metrics_res)
                    logger.info(f"Metrics (Avg): {metrics_avg}")

                    # 绘制指标折线图
                    plot_metrics = ["ACC", "NMI", "Purity", "AR", "RI", "MI", "HI", "F-Score", "Precision", "Recall", "Entropy", "SDCS", "RME", "Bal"]
                    analysis.plot_metric_line(
                        results_list=metrics_res,
                        metrics=plot_metrics,
                        xlabel='Experiment Run ID',
                        ylabel='Score',
                        title=f"{dataset_name} - {exp_id}",
                        save_path=f"{exp_dir / 'line_plot.png'}",
                        show=False
                    )
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
                    "Total_Time": round(elapsed, 4),
                    **params,
                    **metrics_avg
                }
                current_dataset_summary.append(summary_record)
                all_summary.append(summary_record)

                if status == "SUCCESS":
                    acc = metrics_avg.get('ACC', 0)
                    nmi = metrics_avg.get('NMI', 0)
                    ari = metrics_avg.get('AR', 0)
                    # 简单打印进度
                    print(f"  - {exp_id}: ACC={acc:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}  Time={elapsed:.4f}s")

            # =================================================================
            # 【在此处添加代码】: 循环结束后，保存当前数据集的 Summary
            # =================================================================
            if current_dataset_summary:
                save_path = ds_output_dir / f"{dataset_name}_summary.csv"
                pd.DataFrame(current_dataset_summary).to_csv(save_path, index=False)
                print(f"  >> Dataset summary saved: {save_path}")
            # =================================================================

        # 5. 保存总汇总表
        if all_summary:
            df = pd.DataFrame(all_summary)
            df.to_csv(self.output_dir / "grid_summary.csv", index=False)
            print(f"\nAll Done. Summary saved to {self.output_dir / 'grid_summary.csv'}")

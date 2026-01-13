import os
import shutil

import numpy as np
from numpy.random import randint
from sklearn import metrics
import warnings

from .utils import get_threshold, get_elbow, adjust_modular_partition
from .generate_node import generate_brainnet_node
from ....generators.spectral import spectral
from ....consensus.icsc import icsc
from ....analysis.plot import plot_coassociation_heatmap

warnings.filterwarnings("ignore", message="Array is not symmetric, and will be converted")


def ensure_dir(d):
    # Delete previous results
    if os.path.exists(d):
        shutil.rmtree(d)

    # Create results folder if it doesn't exist
    if not os.path.exists(d):
        print("Creating results folder: " + d)
        os.makedirs(d)


def compute_matrix_for_elbow(BPs):
    """Local helper function: used to quickly compute matrix passed to get_elbow"""
    n_samples, n_base = BPs.shape
    sim_mat = np.zeros((n_samples, n_samples))
    for i in range(n_base):
        labels = BPs[:, i]
        match_mat = (labels[:, None] == labels[None, :]).astype(float)
        sim_mat += match_mat
    sim_mat /= n_base
    return sim_mat


def single_multiple_run(params):
    """
    Core processing logic for a single Run
    Note: save_dir is now passed via params, no longer relying on global variables
    """
    (run_id, directory, percent_threshold, individuals_list,
     max_labels, min_labels, num_nodes, dataset, save_dir, heatmap_format) = params

    num_individuals = len(individuals_list)
    np.random.seed()

    # Run-level seed base
    run_seed_base = randint(1, 10000)

    consensus_cost = np.zeros(num_individuals)

    # ICSC Data Structures
    all_individuals_all_L_k = dict()
    individual_labels = dict()
    subjects_data = dict()

    # --- Phase 1: Initial Modularization ---
    print(f"Run {run_id}: Building pools...")

    for count, individual in enumerate(individuals_list):
        # 1. Load & Preprocess
        graph = np.load(os.path.join(directory, individual))
        threshold = get_threshold(graph, percent_threshold)
        graph = graph * (graph > threshold)
        np.fill_diagonal(graph, 0)
        subjects_data[individual] = graph

        # 2. Generate Pool
        subject_pool = dict()
        for k in range(min_labels, max_labels):
            current_seed = run_seed_base + (count * 100) + k
            BPs = spectral(
                X=graph,
                nClusters=k,
                nPartitions=1,
                seed=current_seed,
                n_init=100,
                affinity='precomputed',
                assign_labels='discretize'
            )
            subject_pool[k] = BPs[:, 0]

        all_individuals_all_L_k[individual] = subject_pool

        # 3. Initial Best Selection
        best_k = get_elbow(str(count), graph, max_labels, min_labels)
        if best_k not in subject_pool:
            best_k = list(subject_pool.keys())[0]

        individual_labels[individual] = subject_pool[best_k]

    # --- Phase 2: Initial Group Consensus ---
    current_labels_list = [individual_labels[ind] for ind in individuals_list]
    BPs_current = np.column_stack(current_labels_list)

    temp_matrix = compute_matrix_for_elbow(BPs_current)
    best_group_k = get_elbow('group', temp_matrix, max_labels, min_labels)

    labels_result, _ = icsc(
        BPs=BPs_current,
        nClusters=int(best_group_k),
        nBase=num_individuals,
        nRepeat=1,
        seed=run_seed_base,
        affinity='precomputed'
    )
    group_consensus_labels = labels_result[0]

    # --- Phase 3: Iterative Refinement ---
    for count, individual in enumerate(individuals_list):
        consensus_cost[count] = metrics.adjusted_mutual_info_score(individual_labels[individual],
                                                                   group_consensus_labels)

    consensus_cost_threshold = 0.01
    iteration = 0
    number_individuals_adjusted = num_individuals

    # Save Iteration 0
    iter0_file = os.path.join(save_dir, f'ICSC_group_level_iter_{iteration}.csv')
    with open(iter0_file, 'ab') as out_stream:
        np.savetxt(out_stream, [
            np.append(np.array([run_id, np.mean(consensus_cost), 0, np.unique(group_consensus_labels).size]),
                      group_consensus_labels)], delimiter=", ")

    print(f'Run {run_id} - Iter {iteration}: Cost={np.mean(consensus_cost):.4f}, Modules={number_individuals_adjusted}')

    while number_individuals_adjusted > 0:
        # 1. Update Group Labels
        current_labels_list = [individual_labels[ind] for ind in individuals_list]
        BPs_current = np.column_stack(current_labels_list)

        temp_matrix = compute_matrix_for_elbow(BPs_current)
        L = get_elbow('group', temp_matrix, max_labels, min_labels)

        labels_result, _ = icsc(
            BPs=BPs_current,
            nClusters=int(L),
            nBase=num_individuals,
            nRepeat=1,
            seed=run_seed_base + iteration,
            affinity='precomputed'
        )
        group_consensus_labels = labels_result[0]

        # Update Costs
        for count, individual in enumerate(individuals_list):
            consensus_cost[count] = metrics.adjusted_mutual_info_score(individual_labels[individual],
                                                                       group_consensus_labels)

        # 2. Individual Adjustment
        number_individuals_adjusted = 0
        for count, individual in enumerate(individuals_list):
            labels, new_consensus_cost, modify = adjust_modular_partition(
                all_individuals_all_L_k[individual],
                group_consensus_labels,
                individual_labels[individual],
                consensus_cost_threshold,
                consensus_cost[count],
                'Spectral',
                max_labels,
                min_labels
            )
            individual_labels[individual] = labels
            if modify:
                number_individuals_adjusted += 1
            consensus_cost[count] = new_consensus_cost

        iteration += 1
        print(
            f'Run {run_id} - Iter {iteration}: K={L}, Cost={np.mean(consensus_cost):.4f}, Changed Modules={number_individuals_adjusted}')

        # Save Iteration
        iter_file = os.path.join(save_dir, f'ICSC_group_level_iter_{iteration}.csv')
        with open(iter_file, 'ab') as out_stream:
            np.savetxt(out_stream, [np.append(np.array(
                [run_id, np.mean(consensus_cost), number_individuals_adjusted, np.unique(group_consensus_labels).size]),
                group_consensus_labels)], delimiter=", ")

        # Check Convergence
        if number_individuals_adjusted / num_individuals <= 0:
            # Save Final Summary
            final_file = os.path.join(save_dir, 'ICSC_group_level_final_iter.csv')
            with open(final_file, 'ab') as out_stream:
                np.savetxt(out_stream, [np.append(np.array(
                    [run_id, iteration, np.mean(consensus_cost), number_individuals_adjusted,
                     np.unique(group_consensus_labels).size]), group_consensus_labels)], delimiter=", ")

            # Save Final Matrix
            final_matrix = compute_matrix_for_elbow(BPs_current)
            matrix_file = os.path.join(save_dir, f'group_consensus_matrix_run_{run_id}.csv')
            np.savetxt(matrix_file, final_matrix, delimiter=", ")

            # ========================================================================
            # New Logic Start
            # ========================================================================
            # Generate Consensus Heatmap
            if heatmap_format == 'pdf':
                heatmap_save_path = os.path.join(save_dir, 'pdf', f'consensus_heatmap_run_{run_id}.pdf')
            else:
                heatmap_save_path = os.path.join(save_dir, 'png', f'consensus_heatmap_run_{run_id}.png')
            print(f"Run {run_id}: Generating heatmap -> {heatmap_save_path}")
            plot_coassociation_heatmap(
                Y=group_consensus_labels,
                consensus_matrix=final_matrix,
                title=f"Consensus Matrix (Run {run_id}, Iter {iteration}, ARI={np.mean(consensus_cost):.4f})",
                xlabel="Reordered Nodes",
                ylabel="Reordered Nodes",
                save_path=heatmap_save_path,
                show=False
            )

            # Generate Node File
            node_file_path = os.path.join(save_dir, 'node', f'ICSC_result_run_{run_id}.node')
            print(f"Run {run_id}: Generating Node file -> {node_file_path}")
            generate_brainnet_node(
                labels=group_consensus_labels,
                consensus_matrix=final_matrix,
                data_path=directory,
                save_path=node_file_path
            )
            # ========================================================================
            # New Logic End
            # ========================================================================

            # Save Subject Labels
            subject_labels_file = os.path.join(save_dir, f'ICSC_subject_labels_run_{run_id}.csv')
            for count, individual in enumerate(individuals_list):
                with open(subject_labels_file, 'a') as out_stream:
                    np.savetxt(out_stream, [np.append(np.array([individual]), individual_labels[individual])],
                               delimiter=", ", fmt="%s")
            break

    return run_id

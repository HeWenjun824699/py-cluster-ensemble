import os
import numpy as np
from numpy.random import randint
from sklearn import metrics
import warnings

from .utils import get_elbow, adjust_modular_partition
from ....generators.spectral import spectral
from ....consensus.icsc import icsc

warnings.filterwarnings("ignore", message="Array is not symmetric, and will be converted")


def ensure_dir(d):
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except OSError:
            pass


def compute_matrix_for_elbow(BPs):
    """
    Vectorized implementation for calculating Co-association Matrix.
    Used to replace the originally slow double loop accumulation.
    """
    n_samples, n_base = BPs.shape
    sim_mat = np.zeros((n_samples, n_samples))
    # Vectorized calculation: for each base cluster, calculate its connection matrix and accumulate
    for i in range(n_base):
        labels = BPs[:, i]
        # Use broadcasting mechanism to quickly generate connection matrix (1 if node i and j have same label)
        match_mat = (labels[:, None] == labels[None, :]).astype(float)
        sim_mat += match_mat
    sim_mat /= n_base
    return sim_mat


def single_subject_run(params):
    """
    Process ICSC flow for multiple Sessions of a single Subject.

    Args:
        params (tuple): Contains all configuration and data required for the run

    Different from Group Level:
    Here receives sessions_data (dict) instead of file path list.
    """
    (run_id, percent_threshold, session_ids, sessions_data,
     max_labels, min_labels, num_nodes, dataset, save_dir) = params

    num_sessions = len(session_ids)
    np.random.seed()

    # Run-level seed base, ensuring controllable randomness for each run
    run_seed_base = randint(1, 10000)

    consensus_cost = np.zeros(num_sessions)

    # Initialize data structures
    all_sessions_all_L_k = dict()  # Stores base clusters for each Session under different K values
    session_labels = dict()  # Stores current best labels for each Session

    # --- Phase 1: Initial Modularization (Generate base cluster pool for each Session) ---
    # Note: In Subject Level original logic, data has been thresholded before passing in,
    # but for safety, if raw data is passed, it can be confirmed again here, or assume data is cleaned.
    # Here assumes matrices in sessions_data are already processed (Pre-thresholded & diagonal zeroed).

    for count, session_id in enumerate(session_ids):
        graph = sessions_data[session_id]

        # 1. Generate Pool (Generate clustering results for k=min...max for this Session)
        session_pool = dict()
        for k in range(min_labels, max_labels):
            # Unique seed generation strategy: RunID + SessionID + K, ensuring absolute reproducibility
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
            session_pool[k] = BPs[:, 0]

        all_sessions_all_L_k[session_id] = session_pool

        # 2. Initial Best Selection (Select initial best K for this Session)
        # Use this Session's own matrix to calculate Elbow
        best_k = get_elbow(str(count), graph, max_labels, min_labels)

        # Boundary protection
        if best_k not in session_pool:
            if best_k < min_labels: best_k = min_labels
            if best_k >= max_labels: best_k = max_labels - 1
            if best_k not in session_pool:  # Fallback
                best_k = list(session_pool.keys())[0]

        session_labels[session_id] = session_pool[best_k]

    # --- Phase 2: Initial Subject Consensus (Initialize subject consensus) ---
    current_labels_list = [session_labels[sid] for sid in session_ids]
    BPs_current = np.column_stack(current_labels_list)

    # Use vectorized method to compute consensus matrix for Elbow determination
    temp_matrix = compute_matrix_for_elbow(BPs_current)
    best_subject_k = get_elbow('subject_consensus', temp_matrix, max_labels, min_labels)

    # Run ICSC core algorithm to get initial consensus labels
    labels_result, _ = icsc(
        BPs=BPs_current,
        nClusters=int(best_subject_k),
        nBase=num_sessions,
        nRepeat=1,
        seed=run_seed_base,
        affinity='precomputed'
    )
    subject_consensus_labels = labels_result[0]

    # --- Phase 3: Iterative Refinement (Iterative optimization) ---

    # Calculate initial Cost (AMI)
    for count, session_id in enumerate(session_ids):
        consensus_cost[count] = metrics.adjusted_mutual_info_score(
            session_labels[session_id],
            subject_consensus_labels
        )

    consensus_cost_threshold = 0.01
    iteration = 0
    num_sessions_adjusted = num_sessions

    # 保存 Iteration 0
    ensure_dir(save_dir)
    iter0_file = os.path.join(save_dir, 'ICSC_subject_level_iter_' + str(iteration) + '.csv')
    with open(iter0_file, 'ab') as out_stream:
        np.savetxt(out_stream, [
            np.append(np.array([run_id, np.mean(consensus_cost), 0, np.unique(subject_consensus_labels).size]),
                      subject_consensus_labels)], delimiter=", ")

    print(f'Run {run_id} - Iter {iteration}: Cost={np.mean(consensus_cost):.4f}, Modules={num_sessions_adjusted}')

    while num_sessions_adjusted > 0:
        # 1. Update Subject Consensus Labels based on current session labels
        current_labels_list = [session_labels[sid] for sid in session_ids]
        BPs_current = np.column_stack(current_labels_list)

        temp_matrix = compute_matrix_for_elbow(BPs_current)
        L = get_elbow('subject_consensus', temp_matrix, max_labels, min_labels)

        # Run ICSC again to get new consensus
        labels_result, _ = icsc(
            BPs=BPs_current,
            nClusters=int(L),
            nBase=num_sessions,
            nRepeat=1,
            seed=run_seed_base + iteration,  # Change seed with iteration
            affinity='precomputed'
        )
        subject_consensus_labels = labels_result[0]

        # Update Cost
        for count, session_id in enumerate(session_ids):
            consensus_cost[count] = metrics.adjusted_mutual_info_score(
                session_labels[session_id],
                subject_consensus_labels
            )

        # 2. Individual Session Adjustment (Adjust selection for each Session)
        num_sessions_adjusted = 0
        for count, session_id in enumerate(session_ids):
            # In all candidate clustering results (session_pool) of this Session,
            # find one most similar to current subject_consensus_labels (highest AMI)
            labels, new_consensus_cost, modify = adjust_modular_partition(
                all_sessions_all_L_k[session_id],
                subject_consensus_labels,
                session_labels[session_id],
                consensus_cost_threshold,
                consensus_cost[count],
                'Spectral',
                max_labels,
                min_labels
            )
            session_labels[session_id] = labels
            if modify:
                num_sessions_adjusted += 1
            consensus_cost[count] = new_consensus_cost

        iteration += 1
        print(
            f'Run {run_id} - Iter {iteration}: K={L}, Cost={np.mean(consensus_cost):.4f}, Changed Modules={num_sessions_adjusted}')

        # Save intermediate results
        iter_file = os.path.join(save_dir, 'ICSC_subject_level_iter_' + str(iteration) + '.csv')
        with open(iter_file, 'ab') as out_stream:
            np.savetxt(out_stream, [np.append(np.array(
                [run_id, np.mean(consensus_cost), num_sessions_adjusted, np.unique(subject_consensus_labels).size]),
                subject_consensus_labels)], delimiter=", ")

        # Check convergence
        if num_sessions_adjusted / num_sessions <= 0:
            # Save Final Summary
            final_file = os.path.join(save_dir, 'ICSC_subject_level_final_iter.csv')
            with open(final_file, 'ab') as out_stream:
                np.savetxt(out_stream, [np.append(np.array(
                    [run_id, iteration, np.mean(consensus_cost), num_sessions_adjusted,
                     np.unique(subject_consensus_labels).size]), subject_consensus_labels)], delimiter=", ")

            # Save Final Consensus Matrix
            final_matrix = compute_matrix_for_elbow(BPs_current)
            matrix_file = os.path.join(save_dir, 'subject_consensus_matrix_run_' + str(run_id) + '.csv')
            np.savetxt(matrix_file, final_matrix, delimiter=", ")

            # Save Session Labels
            session_labels_file = os.path.join(save_dir, 'ICSC_session_labels_run_' + str(run_id) + '.csv')
            for count, session_id in enumerate(session_ids):
                with open(session_labels_file, 'a') as out_stream:
                    np.savetxt(out_stream, [np.append(np.array([session_id]), session_labels[session_id])],
                               delimiter=", ", fmt="%s")
            break

    return run_id

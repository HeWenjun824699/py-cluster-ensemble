import pce


# multiple_subjects
pce.applications.icsc_mul_application(
    dataset='multiple_subjects',
    data_directory='./data/multiple_subjects',
    save_dir='./outputs/multiple_subjects',
    num_nodes=264,
    num_runs=1,
    max_labels=21,
    min_labels=5,
    percent_threshold=100,
    heatmap_format='png'
)

# subject_sessions
pce.applications.icsc_sub_application(
    dataset='subject_sessions',
    data_directory='./data/subject_sessions',
    save_dir='./outputs/subject_level_results',
    num_nodes=264,
    max_labels=21,
    min_labels=5,
    percent_threshold=100,
    heatmap_format='png'
)

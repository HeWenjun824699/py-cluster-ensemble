import pce


# main
pce.applications.dcc_application(
    input_path='./dataset',
    output_path='./outputs',
    input_dim=26,
    hidden_dims=range(3, 13),
    epoch=5,
    seed=2026
)

# pce.applications.dcc_cluster_transfer(
#     csv_path='./sample/cluster_transfer.csv',
#     output_dir='./outputs/sample'
# )

# pce.applications.dcc_relative_risk(
#     csv_path='./sample/rr.csv',
#     output_dir='./outputs/sample'
# )

# pce.applications.dcc_comorbidity_bubble(
#     csv_path='./sample/com_iv.csv',
#     output_dir='./outputs/sample'
# )

# pce.applications.dcc_KDIGO_circlize(
#     csv_path='./sample/KDIGO_circlize.csv',
#     output_dir='./outputs/sample'
# )

# pce.applications.dcc_survival_analysis(
#     csv_path='./sample/survival.csv',
#     output_dir='./outputs/sample'
# )

# pce.applications.dcc_KDIGO_dynamic(
#     csv_path='./sample/KDIGO_dynamic.csv',
#     output_dir='./outputs/sample'
# )

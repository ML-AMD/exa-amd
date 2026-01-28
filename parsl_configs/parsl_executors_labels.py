#: Executor label used for structure generation tasks.
GENERATE_EXECUTOR_LABEL = "generate_executor_label"

#: Executor label used for structure selection tasks.
SELECT_EXECUTOR_LABEL = "select_executor_label"

#: Executor label used for CGCNN prediction tasks.
CGCNN_EXECUTOR_LABEL = "cgcnn_executor_label"

#: Executor label used for all VASP-related Parsl tasks.
VASP_EXECUTOR_LABEL = "vasp_executor_label"

#: Executor label for all calculcations and scripts involved in the post-processing
POSTPROCESSING_LABEL = "post_processing_label"

#: Executor label for the MLIP relaxation step
MLIP_RELAXATION_EXECUTOR_LABEL = "mlip_relax_label"

#: Executor label for the ML ehull step (used only in combination with MLIP relaxation)
EHULL_ML_PARALLEL_EXECUTOR_LABEL = "ehull_ml_parallel_label"

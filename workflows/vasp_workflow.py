from tools.config_labels import ConfigKeys as CK
from workflows.core import Workflow
from workflows.registry import register_workflow

from workflows.steps import (
    GenerateStructuresStep,
    CgcnnStep,
    SelectStructuresStep,
    VaspCalculationsStep,
    PostProcessingStep,
)

@register_workflow("vasp")
class VaspWorkflow(Workflow):
    """
    Run the full VASP-based materials discovery workflow.

    Consists of the following task-based steps (with a dependency between each step):

    1. **Structure Generation**
       :func:`~parsl_tasks.gen_structures.generate_structures`

    2. **CGCNN Prediction**
       :func:`~parsl_tasks.cgcnn.run_cgcnn`.

    3. **Structure Selection**
       :func:`~parsl_tasks.cgcnn.select_structures`.

    4. **VASP Calculations**
       :func:`~parsl_tasks.vasp.vasp_calculations`

    5. **Post Processing**

    Args:
        config (ConfigManager): The configuration manager that provides runtime parameters,
            paths, and thresholds for each stage of the workflow.

    Side Effects:
        - Creates directories and files under `config[CK.WORK_DIR]`
        - Executes multiple shell commands and external applications

    Raises:
        Exception: If any sub-stage raises an error that is not internally handled.
    """
    def __init__(self, config, dry_run: bool = False):
        steps = [
            GenerateStructuresStep(config),
            CgcnnStep(config),
            SelectStructuresStep(config, CK.SELECT_STRUCT_OUTPUT),
            VaspCalculationsStep(config),
            PostProcessingStep(config),
        ]
        super().__init__(config=config, steps=steps, dry_run=dry_run)
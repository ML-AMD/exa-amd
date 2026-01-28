from tools.config_labels import ConfigKeys as CK
from workflows.core import Workflow
from workflows.registry import register_workflow
from tools.logging_config import amd_logger

from workflows.steps import (
    GenerateStructuresStep,
    CgcnnStep,
    SelectStructuresStep,
    MLIPRelaxationStep,
    EhullMLParallel,
    VaspCalculationsStep,
    PostProcessingStep,
)

@register_workflow("mlip")
class MLIPWorkflow(Workflow):
    def __init__(self, config, dry_run: bool = False):
        steps = [
            GenerateStructuresStep(config),
            CgcnnStep(config),
            SelectStructuresStep(config, CK.SELECT_STRUCT_OUTPUT_0, 10000, 40000),
            MLIPRelaxationStep(config),
            EhullMLParallel(config),
            VaspCalculationsStep(config, run_mlip_post_processing=True),
            PostProcessingStep(config),
        ]
        super().__init__(config=config, steps=steps, dry_run=dry_run)

    def pre_check(self):
        if not self.config[CK.POST_PROCESSING_OUT_DIR] or not self.config[CK.MPRester_API_KEY]:
            amd_logger.critical("MLIP workflow requires POST_PROCESSING_OUT_DIR and MPRester API key")

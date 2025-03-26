import logging
import parsl

from tools.config_manager import ConfigManager
from tools.logging_config import configure_logging
from parsl_configs.parsl_config_registry import get_parsl_config
from workflows.vasp_based import run_workflow

if __name__ == '__main__':

    # load global config
    config = ConfigManager()

    # load parsl config
    parsl.load(get_parsl_config(config))

    # configure logging
    configure_logging(config["output_level"])

    # run the workflow
    run_workflow(config)

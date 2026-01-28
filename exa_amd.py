import parsl

from tools.config_manager import ConfigManager
from tools.logging_config import amd_logger
from parsl_configs.parsl_config_registry import get_parsl_config
from tools.config_labels import ConfigKeys as CK
from workflows.registry import get_workflow, available_workflows

def main():
    try:
        # load global config
        config = ConfigManager()

        # load parsl config
        parsl.load(get_parsl_config(config))

        # configure logging
        amd_logger.configure(config[CK.OUTPUT_LEVEL])

        # run the workflow
        wf_cls = get_workflow(config[CK.WORKFLOW_NAME])
        workflow = wf_cls(config=config, dry_run=False)
        workflow.run()

    except Exception as e:
        amd_logger.critical(e)

    finally:
        # cleanup
        try:
            dfk = parsl.dfk()
            if dfk is not None:
                dfk.cleanup()
        except parsl.errors.NoDataFlowKernelError:
            pass
        except Exception as cleanup_err:
            amd_logger.warning(f"Parsl cleanup error: {cleanup_err}")

if __name__ == '__main__':
    main()

# workflows/core.py
from typing import Any, Iterable
from tools.logging_config import amd_logger

class Workflow:
    """
    Workflow runner
    - steps: list of steps
    - dry_run: if True, only logs planned steps
    """
    name: str = "base"

    def __init__(self, config: Any, steps: Iterable[Any], dry_run: bool = False):
        self.config = config
        self.steps = list(steps)
        self.dry_run = dry_run

    def pre_check(self) -> None:
        """override in subclasses"""
        pass

    def run(self) -> None:
        self.pre_check()
        amd_logger.info(f"Run workflow: '{getattr(self, 'name', 'unknown')}'")
        if self.dry_run:
            planned = [s.__class__.__name__ for s in self.steps]
            amd_logger.info(
                f"DRY RUN workflow '{getattr(self, 'name', 'unknown')}', steps: {planned}"
            )
            return

        for step in self.steps:
            step_name = step.__class__.__name__
            amd_logger.debug(f"Running step '{step_name}'")
            step.run()
            amd_logger.debug(f"Finished step '{step_name}'")

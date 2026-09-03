"""Parsl task for running MLIP structure relaxation.

Builds and dispatches the shell command that invokes the packaged
``mlip_relax.py`` script to relax structure files using the bundled MLIP model.
"""

from __future__ import annotations

import os
import shlex
from typing import Sequence

from parsl import bash_app

import ml_models.mlip as mlip_pkg
from parsl_configs.parsl_executors_labels import MLIP_RELAXATION_EXECUTOR_LABEL
from tools.config_labels import ConfigKeys as CK

PathLike = str | os.PathLike


def cmd_mlip_relaxation(
    config: dict,
    file_paths: PathLike | Sequence[PathLike],
) -> str:
    """Build the shell command that runs MLIP relaxation on the given files.

    Changes into the configured working directory, ensures the MLIP log
    directory exists, and assembles the command used to invoke the packaged
    ``mlip_relax.py`` script against the bundled model.

    Parameters
    ----------
    config : dict
        A :class:`~tools.config_manager.ConfigManager` (or a dict with the same
        fields). The following keys are read:

        - ``WORK_DIR`` (str): root working directory for inputs/outputs.

        See :class:`~tools.config_manager.ConfigManager` for full field
        descriptions.
    file_paths : str or os.PathLike or sequence of str or os.PathLike
        One or more structure files to relax. A single path is normalized to a
        one-element list.

    Returns
    -------
    str
        The command string invoking ``mlip_relax.py`` with the model path, log
        directory, and shell-quoted file arguments.

    Raises
    ------
    OSError
        If changing directories or creating the log directory fails.
    """
    os.chdir(config[CK.WORK_DIR])

    # for sanity
    if isinstance(file_paths, (str, os.PathLike)):
        file_paths = [file_paths]

    pkg_dir = os.path.dirname(mlip_pkg.__file__)
    model_path = os.path.join(pkg_dir, "uma-s-1p1.pt")
    mlip_relax_script = os.path.join(pkg_dir, "mlip_relax.py")

    energy_log_dir = os.path.join(config[CK.WORK_DIR], CK.MLIP_LOG_DIR)
    os.makedirs(energy_log_dir, exist_ok=True)
    files_argv = " ".join(shlex.quote(str(p)) for p in file_paths)

    return f"python {mlip_relax_script} {model_path} {energy_log_dir} {files_argv}"


@bash_app(executors=[MLIP_RELAXATION_EXECUTOR_LABEL])
def mlip_relaxation(
    config: dict,
    file_paths: PathLike | Sequence[PathLike],
) -> str:
    """Parsl bash app wrapper around :func:`cmd_mlip_relaxation`.

    Parameters
    ----------
    config : dict
        Configuration passed through to :func:`cmd_mlip_relaxation`.
    file_paths : str or os.PathLike or sequence of str or os.PathLike
        Structure file(s) to relax.

    Returns
    -------
    str
        The command string executed by the Parsl ``bash_app``.
    """
    return cmd_mlip_relaxation(config, file_paths)


import os
import pkgutil
import importlib
import importlib.util
import sys
from pathlib import Path

from tools.logging_config import amd_logger
from tools.config_labels import ConfigKeys as CK

# Global registry mapping a config name to its Parsl config class.
PARSL_CONFIG_REGISTRY = {}


def register_parsl_config(config_name, config_class):
    """
    Register a Parsl config class under a string name.

    Can be called inside any module to make the defined configuration
    discoverable at runtime.

    Once a configuration is registered, it can be selected dynamically at runtime
    using the :class:`~tools.config_manager.ConfigManager`, by setting the
    ``parsl_config`` parameter to the corresponding name.

    Args:
        config_name (str): The unique name that identifies the configuration.
        config_class (type): A callable or class that returns a Parsl `Config` object.
    """
    if config_name in PARSL_CONFIG_REGISTRY:
        prior = PARSL_CONFIG_REGISTRY[config_name]
        p_mod = getattr(prior, "__module__", "<unknown_module>")
        p_qual = getattr(prior, "__qualname__", getattr(prior, "__name__", "<unknown>"))
        p_file = _PARSLCONFIG_ORIGINS.get(config_name, ("?", "?", "?"))[2]

        amd_logger.critical(
            "Duplicate Parsl config name '%s'. Existing: %s.%s (%s). "
            "Duplicate: %s.%s (%s). Config names must be unique.",
            config_name, p_mod, p_qual, p_file, mod, qual, src_file
        )
    PARSL_CONFIG_REGISTRY[config_name] = config_class


def auto_register_configs(configs_dir):
    """
    Automatically discover and import all modules that contain a Parsl Configuration.

    Scans the specified **directory** and detects the calls to :func:`register_parsl_config`,
    so the Parsl configurations can be selected at runtime.

    Args:
        configs_dir (str): Path to a directory containing user Parsl config ``*.py`` files.

    Raises:
        SystemExit: If the specified package cannot be found or loaded.
    """
    if not os.path.isdir(configs_dir):
        raise amd_logger.critical(f"Config directory '{configs_dir}' not found or not a directory.")
    dirpath = Path(configs_dir)
    for py in sorted(dirpath.glob("*.py")):
        if py.name.startswith("_"):
            continue
        modname = f"_exa_amd_parsl_cfg.{py.stem}_{abs(hash(py)) & 0xFFFF:x}"
        spec = importlib.util.spec_from_file_location(modname, py)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[modname] = module
            spec.loader.exec_module(module)

    if not PARSL_CONFIG_REGISTRY:
        amd_logger.critical(f"No Parsl configs were registered after scanning '{configs_dir}'. ")


def get_parsl_config(config):
    """
    Retrieve and instantiate the registered Parsl config by name.

    The configuration name is provided via the ``parsl_config`` field in
    :class:`~tools.config_manager.ConfigManager`.

    Args:
        config (dict): Must contain the key ``parsl_config`` and ``parsl_configs_dir``

    Returns:
        parsl.config.Config: An instance of the selected Parsl configuration.

    Raises:
        SystemExit: If the specified config name is not registered.
    """
    if not PARSL_CONFIG_REGISTRY:
        try:
            configs_dir = config[CK.PARSL_CONFIGS_DIR]
        except KeyError as e:
            raise SystemExit(f"Missing required key '{CK.PARSL_CONFIGS_DIR}' in config.") from e
        auto_register_configs(configs_dir)

    config_name = config[CK.PARSL_CONFIG]
    if config_name not in PARSL_CONFIG_REGISTRY:
        amd_logger.critical(f"Parsl config '{config_name}' is not registered.")
    return PARSL_CONFIG_REGISTRY[config_name](config)

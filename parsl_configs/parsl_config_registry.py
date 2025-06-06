
import pkgutil
import importlib

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
    PARSL_CONFIG_REGISTRY[config_name] = config_class


def auto_register_configs(package_name):
    """
    Automatically discover and import all modules that contains a Parsl Configuration.

    Scans the specified package and detects the calls to :func:`register_parsl_config`,
    so the Parsl configurations can be selected at runtime.

    Args:
        package_name (str)

    Raises:
        SystemExit: If the specified package cannot be found or loaded.
    """
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.submodule_search_locations is None:
        amd_logger.critical(
            f"Could not find package {package_name} for auto-registration.")

    for loader, module_name, is_pkg in pkgutil.iter_modules(
            spec.submodule_search_locations):
        full_module_name = f"{package_name}.{module_name}"
        importlib.import_module(full_module_name)


def get_parsl_config(config):
    """
    Retrieve and instantiate the registered Parsl config by name.

    If the registry is empty, it auto-imports all modules
    from the default `parsl_configs` package using :func:`auto_register_configs`.

    The configuration is provided via the ``parsl_config`` field in
    :class:`~tools.config_manager.ConfigManager`.

    Args:
        config (dict): Must contain the key ``parsl_config``

    Returns:
        parsl.config.Config: An instance of the selected Parsl configuration.

    Raises:
        SystemExit: If the specified config name is not registered.
    """
    if not PARSL_CONFIG_REGISTRY:
        auto_register_configs("parsl_configs")

    config_name = config[CK.PARSL_CONFIG]
    if config_name not in PARSL_CONFIG_REGISTRY:
        amd_logger.critical(f"Parsl config '{config_name}' is not registered.")
    return PARSL_CONFIG_REGISTRY[config_name](config)

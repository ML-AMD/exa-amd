import logging
import pkgutil
import importlib

# Global registry mapping a config name to its Parsl config class.
PARSL_CONFIG_REGISTRY = {}


def register_parsl_config(config_name, config_class):
    """
    Register a Parsl config.
    """
    PARSL_CONFIG_REGISTRY[config_name] = config_class


def auto_register_configs(package_name):
    """
    Automatically import all modules in the given package so that
    their registration code is executed.
    """
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.submodule_search_locations is None:
        logging.critical(
            f"Could not find package {package_name} for auto-registration.")

    for loader, module_name, is_pkg in pkgutil.iter_modules(spec.submodule_search_locations):
        full_module_name = f"{package_name}.{module_name}"
        importlib.import_module(full_module_name)


def get_parsl_config(config):
    """
    Retrieve and instantiate the Parsl config.
    """
    if not PARSL_CONFIG_REGISTRY:
        auto_register_configs("parsl_configs")

    config_name = config["parsl_config"]
    if config_name not in PARSL_CONFIG_REGISTRY:
        logging.critical(
            f"Parsl config '{config_name}' is not registered. Registered configs: {list(PARSL_CONFIG_REGISTRY.keys())}"
        )
    return PARSL_CONFIG_REGISTRY[config_name](config)

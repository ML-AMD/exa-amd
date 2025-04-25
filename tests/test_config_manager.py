import sys
import os
import json
import pytest
from tools.config_manager import ConfigManager

#
# Helpers
#


def gen_dummy_value(value, diff=False):
    """
    generate a dummy value
    """
    if isinstance(value, type):
        val_type = value
    else:
        val_type = type(value)

    if val_type == str:
        return "diff_dummy_string" if diff else "dummy_string"
    elif val_type == int:
        return 18 if diff else 17
    elif val_type == float:
        return 18.18 if diff else 17.17
    elif val_type == bool:
        return diff
    else:
        raise ValueError(
            f"Error: Can not generate a dummy value of type {val_type}")


required_config_keys = list(ConfigManager.REQUIRED_PARAMS.keys())
required_dummy_values = [gen_dummy_value(
    val[0]) for val in ConfigManager.REQUIRED_PARAMS.values()]

all_config_keys = required_config_keys + \
    list(ConfigManager.OPTIONAL_PARAMS.keys())
all_dummy_values = required_dummy_values + \
    [gen_dummy_value(val[0]) for val in ConfigManager.OPTIONAL_PARAMS.values()]

valid_config = dict(zip(required_config_keys, required_dummy_values))
complete_config = dict(zip(all_config_keys, all_dummy_values))


#
# Tests
#
def test_valid_config(tmp_path, monkeypatch):
    """
    test that a valid config file is loaded correctly when provided
    """
    config_file = tmp_path / "tmp_config.json"
    config_file.write_text(json.dumps(complete_config))

    # Simulate command-line arguments
    cmd_args = ["python amd.py", "--config", str(config_file)]
    monkeypatch.setattr(sys, "argv", cmd_args)

    config = ConfigManager()

    # Verify the configuration values are as expected.
    for key in all_config_keys:
        if "work_dir" in key:
            assert config[key] == os.path.join(
                complete_config[key], complete_config["elements"])
        else:
            assert config[key] == complete_config[key]


@pytest.mark.parametrize("missing_config_key", required_config_keys)
def test_missing_required_parameters(
        tmp_path, monkeypatch, missing_config_key):
    """
    test that when a required parameter is missing,
    ConfigManager raises an error.
    """
    config_data = valid_config.copy()
    config_data.pop(missing_config_key)
    config_file = tmp_path / "bad_config.json"
    config_file.write_text(json.dumps(config_data))

    cmd_args = ["python amd.py", "--config", str(config_file)]
    monkeypatch.setattr(sys, "argv", cmd_args)

    # Expect a ValueError
    with pytest.raises(ValueError):
        config = ConfigManager()


@pytest.mark.parametrize("config_key", all_config_keys)
def test_command_line_args_override(tmp_path, monkeypatch, config_key):
    """
    Test that command-line arguments override values in the json configuration file.
    """
    config_file = tmp_path / "tmp_config.json"
    config_file.write_text(json.dumps(complete_config))
    override_value = gen_dummy_value(complete_config[config_key], diff=True)

    cmd_args = [
        "python amd.py",
        "--config", str(config_file),
        "--" + config_key, str(override_value)
    ]

    monkeypatch.setattr(sys, "argv", cmd_args)

    config = ConfigManager()

    if "work_dir" in config_key:
        assert config[config_key] != os.path.join(
            complete_config[config_key], complete_config["elements"])
        assert config[config_key] == os.path.join(
            override_value, complete_config["elements"])
    else:
        assert config[config_key] != complete_config[config_key]
        assert config[config_key] == override_value

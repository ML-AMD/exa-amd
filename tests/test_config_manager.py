import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from tools.config_labels import ConfigKeys as CK
from tools.config_manager import ConfigManager, _collect_batch_ids

#
# Helpers
#


def gen_dummy_value(value: Any, diff: bool = False) -> Any:
    """Generate a deterministic dummy value matching a type.

    Parameters
    ----------
    value : Any
        Either a type object or a value whose type is inspected.
    diff : bool, optional
        When ``True``, return a value distinct from the ``False`` case
        (useful for override tests). Defaults to ``False``.

    Returns
    -------
    Any
        A dummy value (``str``, ``int``, ``float`` or ``bool``) matching the
        resolved type.

    Raises
    ------
    ValueError
        If the resolved type has no dummy-value mapping.
    """
    if isinstance(value, type):
        val_type = value
    else:
        val_type = type(value)

    if val_type is str:
        return "diff_dummy_string" if diff else "dummy_string"
    elif val_type is int:
        return 18 if diff else 17
    elif val_type is float:
        return 18.18 if diff else 17.17
    elif val_type is bool:
        return diff
    else:
        raise ValueError(f"Error: Can not generate a dummy value of type {val_type}")


required_config_keys = list(ConfigManager.REQUIRED_PARAMS.keys())
required_dummy_values = [
    gen_dummy_value(val[0]) for val in ConfigManager.REQUIRED_PARAMS.values()
]

all_config_keys = required_config_keys + list(ConfigManager.OPTIONAL_PARAMS.keys())
all_dummy_values = required_dummy_values + [
    gen_dummy_value(val[0]) for val in ConfigManager.OPTIONAL_PARAMS.values()
]

valid_config = dict(zip(required_config_keys, required_dummy_values, strict=True))
complete_config = dict(zip(all_config_keys, all_dummy_values, strict=True))


#
# Tests
#
def test_valid_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete, valid config file loads with all values applied.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to patch ``sys.argv``.
    """
    config_file = tmp_path / "tmp_config.json"
    config_file.write_text(json.dumps(complete_config))

    # Simulate command-line arguments
    cmd_args = ["python exa_amd.py", "--config", str(config_file)]
    monkeypatch.setattr(sys, "argv", cmd_args)

    config = ConfigManager()

    # Verify the configuration values are as expected.
    for key in all_config_keys:
        if CK.WORK_DIR in key:
            assert config[key] == os.path.join(
                complete_config[key], complete_config[CK.ELEMENTS]
            )
        else:
            assert config[key] == complete_config[key]


@pytest.mark.parametrize("missing_config_key", required_config_keys)
def test_missing_required_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_config_key: str
) -> None:
    """A missing required parameter raises ``ValueError``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to patch ``sys.argv``.
    missing_config_key : str
        The required key omitted from the config for this parametrization.
    """
    config_data = valid_config.copy()
    config_data.pop(missing_config_key)
    config_file = tmp_path / "bad_config.json"
    config_file.write_text(json.dumps(config_data))

    cmd_args = ["python exa_amd.py", "--config", str(config_file)]
    monkeypatch.setattr(sys, "argv", cmd_args)

    # Expect a ValueError
    with pytest.raises(ValueError):
        ConfigManager()


@pytest.mark.parametrize("config_key", all_config_keys)
def test_command_line_args_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_key: str
) -> None:
    """Command-line arguments override values from the JSON config.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to patch ``sys.argv``.
    config_key : str
        The config key being overridden for this parametrization.
    """
    config_file = tmp_path / "tmp_config.json"
    config_file.write_text(json.dumps(complete_config))
    override_value = gen_dummy_value(complete_config[config_key], diff=True)

    cmd_args = [
        "python exa_amd.py",
        "--config",
        str(config_file),
        "--" + config_key,
        str(override_value),
    ]

    monkeypatch.setattr(sys, "argv", cmd_args)

    config = ConfigManager()

    if CK.WORK_DIR in config_key:
        assert config[config_key] != os.path.join(
            complete_config[config_key], complete_config[CK.ELEMENTS]
        )
        assert config[config_key] == os.path.join(
            override_value, complete_config[CK.ELEMENTS]
        )
    else:
        assert config[config_key] != complete_config[config_key]
        assert config[config_key] == override_value


def _write_config(tmp_path: Path, data: Dict[str, Any]) -> Path:
    """Write ``data`` as a JSON config file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory in which to create the file.
    data : dict
        Configuration payload to serialize.

    Returns
    -------
    pathlib.Path
        Path to the written JSON file.
    """
    config_file = tmp_path / "cfg.json"
    config_file.write_text(json.dumps(data))
    return config_file


def test_optional_defaults_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent optional parameters receive their declared defaults.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to patch ``sys.argv``.
    """
    config_file = _write_config(tmp_path, valid_config)
    monkeypatch.setattr(sys, "argv", ["exa_amd.py", "--config", str(config_file)])

    config = ConfigManager()

    for key, (default_val, _) in ConfigManager.OPTIONAL_PARAMS.items():
        assert config[key] == default_val


def test_missing_config_flag_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``--config`` flag causes the parser to exit.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to patch ``sys.argv``.
    """
    monkeypatch.setattr(sys, "argv", ["exa_amd.py"])

    with pytest.raises(SystemExit):
        ConfigManager()


def test_nonexistent_config_file_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing config file path triggers ``sys.exit``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to patch ``sys.argv``.
    """
    missing = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(sys, "argv", ["exa_amd.py", "--config", str(missing)])

    with pytest.raises(SystemExit):
        ConfigManager()


def test_invalid_json_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed JSON in the config file triggers ``sys.exit``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to patch ``sys.argv``.
    """
    config_file = tmp_path / "bad.json"
    config_file.write_text("{ not valid json")
    monkeypatch.setattr(sys, "argv", ["exa_amd.py", "--config", str(config_file)])

    with pytest.raises(SystemExit):
        ConfigManager()


def test_post_processing_requires_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting a post-processing dir without an API key raises ``ValueError``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to patch ``sys.argv``.
    """
    data = valid_config.copy()
    data[CK.POST_PROCESSING_OUT_DIR] = str(tmp_path / "pp")
    data[CK.MPRester_API_KEY] = ""
    config_file = _write_config(tmp_path, data)
    monkeypatch.setattr(sys, "argv", ["exa_amd.py", "--config", str(config_file)])

    with pytest.raises(ValueError):
        ConfigManager()


def test_early_help_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing ``--help`` prints usage and exits before config loading.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to patch ``sys.argv``.
    """
    monkeypatch.setattr(sys, "argv", ["exa_amd.py", "--help"])

    with pytest.raises(SystemExit):
        ConfigManager()


def test_get_json_config_returns_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``get_json_config`` returns the merged configuration dict.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to patch ``sys.argv``.
    """
    config_file = _write_config(tmp_path, valid_config)
    monkeypatch.setattr(sys, "argv", ["exa_amd.py", "--config", str(config_file)])

    config = ConfigManager()
    result = config.get_json_config()

    assert isinstance(result, dict)
    assert result[CK.WORKFLOW_NAME] == valid_config[CK.WORKFLOW_NAME]


def _make_poscar_dir(root: Path, ids: list[int]) -> Path:
    """Create a directory of ``POSCAR_<id>`` files.

    Parameters
    ----------
    root : pathlib.Path
        Parent directory in which to create the structure directory.
    ids : list[int]
        POSCAR identifiers to create.

    Returns
    -------
    pathlib.Path
        The created structure directory.
    """
    structure_dir = root / "structures"
    structure_dir.mkdir()
    for i in ids:
        (structure_dir / f"POSCAR_{i}").write_text("POSCAR")
    return structure_dir


def _make_run_dir(root: Path, done_ids: list[int], pending_ids: list[int]) -> Path:
    """Create numbered VASP run directories with optional ``DONE`` markers.

    Parameters
    ----------
    root : pathlib.Path
        Parent directory for the VASP work directory.
    done_ids : list[int]
        Run ids that should contain a ``DONE`` marker.
    pending_ids : list[int]
        Run ids that exist without a ``DONE`` marker.

    Returns
    -------
    pathlib.Path
        The created VASP work directory.
    """
    vasp_dir = root / "vasp"
    vasp_dir.mkdir()
    for i in done_ids:
        d = vasp_dir / str(i)
        d.mkdir()
        (d / "DONE").write_text("")
    for i in pending_ids:
        (vasp_dir / str(i)).mkdir()
    return vasp_dir


def test_collect_batch_ids_no_poscars_returns_empty(tmp_path: Path) -> None:
    """An empty structure directory yields no batch ids.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    """
    structure_dir = tmp_path / "structures"
    structure_dir.mkdir()
    vasp_dir = tmp_path / "vasp"
    vasp_dir.mkdir()

    assert _collect_batch_ids(str(vasp_dir), str(structure_dir), -1) == []


def test_collect_batch_ids_all_new(tmp_path: Path) -> None:
    """With no existing runs, all POSCAR ids are returned sorted.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    """
    structure_dir = _make_poscar_dir(tmp_path, [3, 1, 2])
    vasp_dir = tmp_path / "vasp"
    vasp_dir.mkdir()

    assert _collect_batch_ids(str(vasp_dir), str(structure_dir), -1) == [1, 2, 3]


def test_collect_batch_ids_skips_done_includes_unfinished(tmp_path: Path) -> None:
    """Finished runs are skipped while unfinished ones are rerun.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    """
    structure_dir = _make_poscar_dir(tmp_path, [1, 2, 3])
    vasp_dir = _make_run_dir(tmp_path, done_ids=[1], pending_ids=[2])

    result = _collect_batch_ids(str(vasp_dir), str(structure_dir), -1)

    # id 1 is DONE (skip), id 2 unfinished (rerun), id 3 new.
    assert result == [2, 3]


def test_collect_batch_ids_respects_nstructures_limit(tmp_path: Path) -> None:
    """A positive ``nstructures`` limits the number of returned ids.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    """
    structure_dir = _make_poscar_dir(tmp_path, [1, 2, 3, 4])
    vasp_dir = tmp_path / "vasp"
    vasp_dir.mkdir()

    assert _collect_batch_ids(str(vasp_dir), str(structure_dir), 2) == [1, 2]


def test_collect_batch_ids_removes_stale_potcar(tmp_path: Path) -> None:
    """A stale POTCAR in an unfinished run directory is removed.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory.
    """
    structure_dir = _make_poscar_dir(tmp_path, [1])
    vasp_dir = _make_run_dir(tmp_path, done_ids=[], pending_ids=[1])
    stale_potcar = vasp_dir / "1" / "POTCAR"
    stale_potcar.write_text("stale")

    result = _collect_batch_ids(str(vasp_dir), str(structure_dir), -1)

    assert result == [1]
    assert not stale_potcar.exists()

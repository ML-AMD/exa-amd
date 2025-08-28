import argparse
import json
import os
import sys
import re
import subprocess
from pathlib import Path
from shutil import copyfileobj

from tools.config_labels import ConfigKeys as CK


def _find_next_vasp_structure(work_dir):
    """Helper: Find what should be the next VASP calculation"""
    # Get current directory contents
    contents = os.listdir(work_dir)

    # Filter only directories and find those that match numeric pattern
    numbered_dirs = []
    for item in contents:
        if os.path.isdir(os.path.join(work_dir, item)):
            # Try to extract a number from the directory name
            match = re.search(r'^\d+', item)
            if match:
                numbered_dirs.append(int(match.group()))

    if not numbered_dirs:
        return 1  # If no numbered directories exist, start with 1

    # Find the highest number and add 1
    next_number = max(numbered_dirs) + 1
    return next_number


class ConfigManager:
    """
    Manages configuration settings loaded from the JSON config file and optionally overridden
    by command-line arguments.

    This class ensures that required parameters are present and valid, while applying
    defaults for optional settings.

    """
    # required arguments: must exist in JSON config or be provided as cmd line
    REQUIRED_PARAMS = {
        CK.CMS_DIR: (str, "Path to the CMS directory (required)."),
        CK.VASP_STD_EXE: (str, "VASP executable (required)."),
        CK.WORK_DIR: (str, "Path to a work directory used for generating and selecting all the structures (required)."),
        CK.VASP_WORK_DIR: (str, "Path to a work directory for VASP-specific operations (required)."),
        CK.POT_DIR: (str, "Path to the PAW potentials directory containing kinetic energy densities for meta-GGA calculations (required)."),
        CK.OUTPUT_FILE: (str, "Output file name for storing the result of the VASP calculations (required)."),
        CK.ELEMENTS: (str, "Elements, e.g. 'Ce-Co-B' (required)."),
        CK.PARSL_CONFIG: (
            str, "Parsl config name, previously registered (required)."),
        CK.INITIAL_STRS: (
            str, "Path to the directory that containts the initial crystal structures.")
    }

    # optional arguments: if absent, assign defaults.
    OPTIONAL_PARAMS = {
        CK.EF_THR: (-0.2, "A formation energy threshold used for selecting the structures, after the CGCNN prediction."),
        CK.NUM_WORKERS: (128, "Number of threads used for generating, predicting and selecting the structures."),
        CK.BATCH_SIZE: (256, "Batch size for CGCNN."),
        CK.VASP_NNODES: (1, "Number of nodes used for VASP calculations."),
        CK.VASP_NTASKS_PER_RUN: (1, "Number of MPI processes per VASP calculation (useful for CPU-only Parsl configurations)."),
        CK.NUM_STRS: (-1, "Number of structures to be processed with VASP. (-1 means all)."),
        CK.VASP_TIMEOUT: (1800, "Max walltime in seconds for a VASP calculation."),
        CK.VASP_NSW: (100, "VASP NSW: gives the number of steps in all molecular dynamics runs."),
        CK.CPU_ACCOUNT: ("", "The cpu account name on the current machine (forwarded to the workload manager)."),
        CK.GPU_ACCOUNT: ("", "The gpu account name on the current machine (forwarded to the workload manager)."),
        CK.OUTPUT_LEVEL: ("INFO", "Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"),
        CK.POST_PROCESSING_OUT_DIR: ("", "A full path to a directory that will contain all the post-processing results. If not set, the post-processing step will be skipped."),
        CK.MPRester_API_KEY: ("", f"An API key for accessing the MP data (https://docs.materialsproject.org). Required if --{CK.POST_PROCESSING_OUT_DIR} is set. "),
        CK.HULL_ENERGY_THR: (
            0.1, "Maximum Ehull (eV/atom) to display for metastable phases"),
        CK.GEN_STRUCTURES_NNODES: (1, "Number of nodes used for the pre-processing phases")
    }

    CONFIG_HELP_MSG = "Path to the JSON configuration file (required)."
    HELP_DESCRIPTION = "Override JSON config fields with command line arguments."

    def __init__(self):
        """
        Load the json config and apply the cmd line arguments.
        Check if all the required arguments were provided.
        """

        self._early_help()
        # Preliminary parser for -config (read only the JSON path)
        config_parser = argparse.ArgumentParser(add_help=False)
        config_parser.add_argument(
            f"--{CK.CONFIG_FILE}",
            type=str,
            default=None,
            help=self.CONFIG_HELP_MSG
        )
        config_args, remaining_args = config_parser.parse_known_args()
        self.config_path = config_args.config

        # Load JSON config
        if self.config_path is None:
            config_parser.error(
                "Please provide a json configuration file (e.g., --config my_config.json)")
        else:
            if not os.path.exists(self.config_path):
                print(f"Config file {self.config_path} not found. Aborting.")
                sys.exit(1)
            try:
                with open(self.config_path, "r") as f:
                    self.config = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON config: {e}")
                sys.exit(1)

        parser = argparse.ArgumentParser(
            parents=[config_parser],
            description=self.HELP_DESCRIPTION
        )

        # Loop over REQUIRED_PARAMS
        for key, (arg_type, help_text) in self.REQUIRED_PARAMS.items():
            parser.add_argument(
                f"--{key}",
                type=arg_type,
                default=None,  # We'll check existence later
                help=help_text
            )

        # Loop over OPTIONAL_PARAMS
        for key, (default_val, help_text) in self.OPTIONAL_PARAMS.items():
            arg_type = type(default_val)
            parser.add_argument(
                f"--{key}",
                default=None,  # We'll assign defaults ourselves if needed
                type=arg_type,
                help=f"{help_text} (default='{default_val}')."
            )

        args = parser.parse_args(remaining_args)

        for arg_name in vars(args):
            value = getattr(args, arg_name)
            if value is not None:
                old_val = self.config.get(arg_name)
                self.config[arg_name] = value
                if old_val is not None:
                    print(f"Overriding '{arg_name}': {old_val} -> {value}")

        # Ensure all required params exist post-merge
        for key in self.REQUIRED_PARAMS.keys():
            if key not in self.config:
                raise ValueError(f"Error: Missing required argument '{key}'.")

        # Assign defaults for optional params
        for key, (default_val, _) in self.OPTIONAL_PARAMS.items():
            if key not in self.config:
                self.config[key] = default_val

        if self.config[CK.POST_PROCESSING_OUT_DIR] and not self.config[CK.MPRester_API_KEY]:
            raise ValueError(f"Error: Missing required argument '{self.config[MPRester_API_KEY]}'.")

        # Create/Update directories
        work_dir = os.path.join(
            self.config[CK.WORK_DIR], self.config[CK.ELEMENTS])
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        vasp_work_dir = os.path.join(
            self.config[CK.VASP_WORK_DIR], self.config[CK.ELEMENTS])
        if not os.path.exists(vasp_work_dir):
            os.makedirs(vasp_work_dir)

        self.config[CK.WORK_DIR] = work_dir
        self.config[CK.VASP_WORK_DIR] = vasp_work_dir

    def _early_help(self):
        if "-h" in sys.argv or "--help" in sys.argv:
            parser = argparse.ArgumentParser(
                description=self.HELP_DESCRIPTION
            )

            parser.add_argument(
                f"--{CK.CONFIG_FILE}",
                type=str,
                default=None,
                help=self.CONFIG_HELP_MSG
            )

            # Loop over REQUIRED_PARAMS
            for key, (arg_type, help_text) in self.REQUIRED_PARAMS.items():
                parser.add_argument(
                    f"--{key}",
                    type=arg_type,
                    default=None,  # We'll check existence later
                    help=help_text
                )

            # Loop over OPTIONAL_PARAMS
            for key, (default_val, help_text) in self.OPTIONAL_PARAMS.items():
                arg_type = type(default_val)
                parser.add_argument(
                    f"--{key}",
                    default=None,  # We'll assign defaults ourselves if needed
                    type=arg_type,
                    help=f"{help_text} (default='{default_val}')."
                )
            parser.parse_args()

    def setup_vasp_calculations(self):
        """
        Calculate nstart and nend for VASP calculations.
        All structures in [nstart, nend) will be run.
        """
        work_dir = self.config[CK.WORK_DIR]
        structure_dir = os.path.join(self.config[CK.WORK_DIR], "new")
        structure_files = [
            f for f in os.listdir(structure_dir) if f.startswith("POSCAR_")
        ]
        total_num_structures = len(structure_files)
        num_strs = self.config[CK.NUM_STRS]
        nstart = _find_next_vasp_structure(self.config[CK.VASP_WORK_DIR])

        nend = total_num_structures + 1
        if num_strs != -1:
            nend = min(nstart + num_strs, nend)

        self.config["nstart"] = nstart
        self.config["nend"] = nend

        # create the POTCAR file
        elements = self.config[CK.ELEMENTS].split('-')
        nb_of_elements = len(elements)

        if nb_of_elements < 3 or nb_of_elements > 4:
            amd_logger.critical(
                f"exa-AMD only supports ternary and quaternary systems")

        POTDIR = self.config[CK.POT_DIR]
        # create the POTCAR file
        potcar_paths = [Path(POTDIR) / el / "POTCAR" for el in elements]
        out_path = Path(work_dir) / "POTCAR"

        with out_path.open("wb") as out:
            for p in potcar_paths:
                with p.open("rb") as inp:
                    copyfileobj(inp, out, length=1024 * 1024)

    def _create_potcar(self):
        POTDIR = self.config[CK.POT_DIR]
        ele1, ele2, ele3 = self.config[CK.ELEMENTS].split("-")
        potcar_paths = [f"{POTDIR}/{el}/POTCAR" for el in (ele1, ele2, ele3)]

        with open(f"{work_dir}/POTCAR", "wb") as outfile:
            for path in potcar_paths:
                with open(path, "rb") as infile:
                    outfile.write(infile.read())

    def get_json_config(self):
        """Return the JSON configuration."""
        return self.config

    def __getitem__(self, key):
        """Allow dictionary-like access to configuration items."""
        return self.config[key]

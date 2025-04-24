import argparse
import json
import os
import sys
import re


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
    # required arguments: must exist in JSON config or be provided as cmd line
    REQUIRED_PARAMS = {
        "cms_dir": (str, "Path to the CMS directory (required)."),
        "vasp_std_exe": (str, "Path to the VASP executable (required)."),
        "work_dir": (str, "Root working directory (required)."),
        "vasp_work_dir": (str, "Working directory for VASP-specific operations (required)."),
        "pot_dir": (str, "Path to potpaw (required)."),
        "output_file": (str, "Output file path (required)."),
        "elements": (str, "Elements, e.g. 'Ce-Co-B' (required)."),
        "parsl_config": (str, "Parsl config name, previously registered (required).")
    }

    # optional arguments: if absent, assign defaults.
    OPTIONAL_PARAMS = {
        "ef_thr": (-0.2, "ef threshold."),
        "num_workers": (128, "Number of OpenMP threads."),
        "batch_size": (256, "Batch size for CGCNN."),
        "vasp_nnodes": (1, "Number of nodes used for VASP calculations."),
        "num_strs": (-1, "Number of structures to process (-1 means all)."),
        "vasp_timeout": (1800, "Max walltime in seconds for a vasp calculation."),
        "force_conv": (100, "Force convergence threshold."),
        "output_level": ("INFO", "Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"),
    }

    def __init__(self):
        """
        Load the json config and apply the cmd line arguments.
        Check if all the required arguments were provided.
        """
        # Preliminary parser for -config (read only the JSON path)
        config_parser = argparse.ArgumentParser(add_help=False)
        config_parser.add_argument(
            "--config",
            type=str,
            default="configs/chicoma.json",
            help="Path to the JSON configuration file (default: configs/chicoma.json)"
        )
        config_args, remaining_args = config_parser.parse_known_args()
        self.config_path = config_args.config

        # Load JSON config
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
            description="Override JSON config fields with command line arguments."
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

        # Create/Update directories
        work_dir = os.path.join(
            self.config["work_dir"], self.config["elements"])
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        vasp_work_dir = os.path.join(
            self.config["vasp_work_dir"], self.config["elements"])
        if not os.path.exists(vasp_work_dir):
            os.makedirs(vasp_work_dir)

        self.config["work_dir"] = work_dir
        self.config["vasp_work_dir"] = vasp_work_dir

    def setup_vasp_calculations(self):
        """
        Calculate nstart and nend for VASP calculations.
        All structures in [nstart, nend) will be run
        """
        work_dir = self.config["work_dir"]
        structure_dir = os.path.join(self.config["work_dir"], "new")
        structure_files = [
            f for f in os.listdir(structure_dir) if f.startswith("POSCAR_")
        ]
        total_num_structures = len(structure_files)
        num_strs = self.config["num_strs"]
        nstart = _find_next_vasp_structure(self.config["vasp_work_dir"])

        nend = total_num_structures + 1
        if num_strs != -1:
            nend = min(nstart + num_strs, nend)

        self.config["nstart"] = nstart
        self.config["nend"] = nend

        # create the POTCAR file
        POTDIR = self.config["pot_dir"]
        ele1, ele2, ele3 = self.config["elements"].split("-")
        potcar_command = (
            f"cat {POTDIR}/{ele1}/POTCAR "
            f"{POTDIR}/{ele2}/POTCAR "
            f"{POTDIR}/{ele3}/POTCAR "
            f"> {work_dir}/POTCAR"
        )
        os.system(potcar_command)

    def get_json_config(self):
        """Return the JSON configuration."""
        return self.config

    def __getitem__(self, key):
        """Allow dictionary-like access to configuration items."""
        return self.config[key]

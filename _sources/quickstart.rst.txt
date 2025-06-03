Quickstart
==========

Prerequisites
-------------
**exa-AMD requires:**

- python = 3.12
- scikit-learn
- pytorch
- torchvision
- pymatgen
- parsl
- pytest

**Additionally:**

- Ensure you have a working VASP installation
- Ensure you have prepared the initial crystal structures in the Crystallographic Information File (CIF) format

.. _installation:

Installation
------------
- Ensure you have Conda installed.
- Install the required packages:

.. code-block:: bash

    conda env create -f amd_env.yml


Run the tests
-------------

.. code-block:: bash

    conda activate amd_env
    pytest


Using a JSON Configuration File
-------------------------------

The recommended way to configure exa-AMD is through a JSON configuration file.
It specifies all the required and optional parameters for running the workflow.

Here is an example configuration file for the Perlmutter system:

.. code-block:: json

    {
        "cms_dir": "~/exa-amd/cms_dir",
        "vasp_std_exe": "vasp_std",
        "work_dir": "/pscratch/sd/<user_name>/work_dir",
        "vasp_work_dir": "/pscratch/sd/<user_name>/vasp_work_dir",
        "pot_dir": "~/potpaw_PBE",
        "output_file": "vasp_results.csv",
        "num_workers": 128,
        "batch_size": 256,
        "ef_thr": -0.2,
        "force_conv": 100,
        "vasp_timeout": 1800,
        "num_strs": 10,
        "elements": "Na-B-C",
        "vasp_nnodes": 1,
        "parsl_config": "perlmutter"
    }

You can create multiple configuration files for different systems, workloads, or experiments.

Command-line Usage
------------------

You can override any field from the JSON configuration using command-line arguments:

.. code-block:: bash

    python amd.py --help

    usage: exa_amd.py [-h] [--config CONFIG] [--cms_dir CMS_DIR] [--vasp_std_exe VASP_STD_EXE] [--work_dir WORK_DIR] [--vasp_work_dir VASP_WORK_DIR] [--vasp_pot_dir VASP_POT_DIR] [--vasp_output_file VASP_OUTPUT_FILE] [--elements ELEMENTS] [--parsl_config PARSL_CONFIG]
                    [--formation_energy_threshold FORMATION_ENERGY_THRESHOLD] [--num_workers NUM_WORKERS] [--cgcnn_batch_size CGCNN_BATCH_SIZE] [--vasp_nnodes VASP_NNODES] [--vasp_ntasks_per_run VASP_NTASKS_PER_RUN] [--vasp_nstructures VASP_NSTRUCTURES] [--vasp_timeout VASP_TIMEOUT]
                    [--vasp_force_conv VASP_FORCE_CONV] [--output_level OUTPUT_LEVEL]

    Override JSON config fields with command line arguments.

    options:
    -h, --help            show this help message and exit
    --config CONFIG       Path to the JSON configuration file (required).
    --cms_dir CMS_DIR     Path to the CMS directory (required).
    --vasp_std_exe VASP_STD_EXE
                            VASP executable (required).
    --work_dir WORK_DIR   Path to a work directory used for generating and selecting all the structures (required).
    --vasp_work_dir VASP_WORK_DIR
                            Path to a work directory for VASP-specific operations (required).
    --vasp_pot_dir VASP_POT_DIR
                            Path to the PAW potentials directory containing kinetic energy densities for meta-GGA calculations (required).
    --vasp_output_file VASP_OUTPUT_FILE
                            Output file name for storing the result of the VASP calculations (required).
    --elements ELEMENTS   Elements, e.g. 'Ce-Co-B' (required).
    --parsl_config PARSL_CONFIG
                            Parsl config name, previously registered (required).
    --formation_energy_threshold FORMATION_ENERGY_THRESHOLD
                            A formation energy threshold used for selecting the structures, after the CGCNN prediction. (default='-0.2').
    --num_workers NUM_WORKERS
                            Number of threads used for generating, predicting and selecting the structures. (default='128').
    --cgcnn_batch_size CGCNN_BATCH_SIZE
                            Batch size for CGCNN. (default='256').
    --vasp_nnodes VASP_NNODES
                            Number of nodes used for VASP calculations. (default='1').
    --vasp_ntasks_per_run VASP_NTASKS_PER_RUN
                            Number of MPI processes per VASP calculation (useful for CPU-only Parsl configurations). (default='1').
    --vasp_nstructures VASP_NSTRUCTURES
                            Number of structures to be processed with VASP. (-1 means all). (default='-1').
    --vasp_timeout VASP_TIMEOUT
                            Max walltime in seconds for a VASP calculation. (default='1800').
    --vasp_force_conv VASP_FORCE_CONV
                            VASP force convergence threshold. (default='100').
    --output_level OUTPUT_LEVEL
                            Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default='INFO').

Quickstart
==========

Prerequisites
-------------
**exa-AMD requires:**

- python >= 3.10
- numpy < 2.0
- scikit-learn >= 1.6.1
- pytorch >= 2.2.2
- torchvision >= 0.17.2
- pymatgen >= 2025.3.10
- parsl >= 2025.3.24
- pytest >= 8.3.5
- sphinx >= 7.1.2
- sphinx_rtd_theme >= 3.0.2
- mp-api >= 0.45.7
- python-ternary >= 1.0.8

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
        "work_dir": "<abs_path_to>/work_dir",
        "cpu_account": "cpu_account",
        "gpu_account": "gpu_account",
        "elements": "Na-B-C",
        "formation_energy_threshold": -0.2,
        "num_workers": 128,
        "initial_structures_dir":"<abs_path_to>/initial_structures",

        "parsl_config": "perlmutter_premium",

        "pre_processing_nnodes": 1,

        "cgcnn_batch_size": 256,

        "vasp_std_exe": "vasp_std",
        "vasp_work_dir": "<abs_path_to>/vasp_work_dir",
        "vasp_pot_dir": "<abs_path_to>/potpaw_PBE",
        "vasp_output_file": "vasp_results.csv",
        "vasp_nstructures": 10,
        "vasp_nsw": 100,
        "vasp_timeout": 1800,
        "vasp_nnodes": 1,

        "hull_energy_threshold": 0.1,
        "post_processing_output_dir": "<abs_path_to>/post_processing_out_dir",
        "mp_rester_api_key": "<MP_RESTER_API_KEY>"
    }

You can create multiple configuration files for different systems, workloads, or experiments.

Command-line Usage
------------------

You can override any field from the JSON configuration using command-line arguments.

.. code-block:: bash

    python exa_amd.py --help
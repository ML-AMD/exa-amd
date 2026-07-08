.. _configuration_reference:

=============================
Configuration Reference Guide
=============================

This page provides an exhaustive reference for all JSON configuration options in **exa-AMD**. Use your browser's search function (CTRL+F or CMD+F) to quickly locate specific parameters.

Overview
========

**exa-AMD** uses JSON configuration files to specify workflow parameters, computational resources, file paths, and runtime settings. The configuration system provides:

- **Required parameters**: Must be provided in the JSON file or via command-line arguments
- **Optional parameters**: Have sensible defaults but can be customized
- **Command-line overrides**: Supported configuration parameters can be overridden using ``--parameter_name value``

Example:

.. code-block:: bash

   exa_amd --config my_config.json --num_workers 256 --vasp_nnodes 4

Path Requirements
-----------------

.. important::

   All directory paths should be **absolute paths**, not relative paths. Using absolute paths ensures consistent behavior across different execution contexts and prevents path resolution issues.

Workflow Selection
------------------

The ``workflow`` parameter determines which workflow implementation to use:

- ``"vasp"``: Standard CGCNN-to-DFT workflow (see :doc:`workflow`)
- ``"mlip"``: MLIP relaxation and hull sorting workflow (see :doc:`mlip_workflow`)

Required Parameters
===================

These parameters must be present in your JSON configuration file or provided via command-line arguments.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Type
     - Description
   * - ``workflow``
     - string
     - Workflow to be run. Available workflows: ``"vasp"``, ``"mlip"``. See :doc:`workflow` and :doc:`mlip_workflow` for workflow-specific details.
   * - ``work_dir``
     - string
     - Absolute path to the work directory used for generating and selecting all structures. A subdirectory named after ``elements`` will be automatically created (e.g., ``work_dir/Na-B-C/``).
   * - ``vasp_work_dir``
     - string
     - Absolute path to the work directory for VASP-specific operations. A subdirectory named after ``elements`` will be automatically created.
   * - ``vasp_std_exe``
     - string
     - Name or path to the VASP executable (e.g., ``"vasp_std"``). Must be accessible in the compute environment where VASP calculations run.
   * - ``vasp_pot_dir``
     - string
     - Absolute path to the directory containing PAW potentials (e.g., ``potpaw_PBE``). Must include kinetic energy densities for meta-GGA calculations. Each element subdirectory should contain a ``POTCAR`` file.
   * - ``vasp_output_file``
     - string
     - Filename for storing VASP calculation results (e.g., ``"vasp_results.csv"``). This file will be created in ``vasp_work_dir/<elements>/``.
   * - ``elements``
     - string
     - Target elements for materials discovery, separated by hyphens. Examples: ``"Ce-Co-B"`` (ternary), ``"Na-B-H-C"`` (quaternary). Only ternary and quaternary systems are supported.
   * - ``initial_structures_dir``
     - string
     - Absolute path to the directory containing initial crystal structures in CIF format. These structures serve as templates for generating hypothetical compounds.
   * - ``parsl_config``
     - string
     - Name of the registered Parsl configuration to use (e.g., ``"perlmutter_premium"``). Must match a configuration registered via ``register_parsl_config()`` in your Parsl configs directory. See :doc:`parsl_config` for details.
   * - ``parsl_configs_dir``
     - string
     - Absolute path to the directory containing Parsl configuration Python files. exa-AMD will automatically discover and register all configs in this directory at runtime.

Optional Parameters
===================

These parameters have default values but can be customized in your JSON configuration file.

General Settings
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 50

   * - Key
     - Type
     - Default
     - Description
   * - ``num_workers``
     - integer
     - 128
     - Number of CPU threads used for structure generation, CGCNN prediction, and structure selection stages. Adjust based on available cores.
   * - ``output_level``
     - string
     - ``"INFO"``
     - Logging level. Valid values: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``. Use ``"DEBUG"`` for detailed troubleshooting.

Compute Resources
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 50

   * - Key
     - Type
     - Default
     - Description
   * - ``cpu_account``
     - string
     - ``""``
     - CPU account name for the workload manager (e.g., Slurm). Required if your HPC system enforces account-based resource allocation. Can also be specified in the Parsl configuration.
   * - ``gpu_account``
     - string
     - ``""``
     - GPU account name for the workload manager (e.g., Slurm). Required if your HPC system enforces account-based resource allocation. Can also be specified in the Parsl configuration.
   * - ``pre_processing_nnodes``
     - integer
     - 1
     - Number of CPU nodes allocated for the structure generation stage. Increase for large-scale structure enumeration.
   * - ``mlip_relax_nnodes``
     - integer
     - 1
     - Number of GPU nodes allocated for MLIP relaxation (used only by the ``mlip`` workflow). Each node should have multiple GPUs as specified by ``gpus_per_node``.
   * - ``gpus_per_node``
     - integer
     - 4
     - Number of GPUs per node (used only by the ``mlip`` workflow). Used to partition MLIP relaxation work across available GPUs.

CGCNN Parameters
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 50

   * - Key
     - Type
     - Default
     - Description
   * - ``formation_energy_threshold``
     - float
     - -0.2
     - Formation energy threshold (eV/atom) for structure selection after CGCNN prediction. Structures with predicted formation energy below this threshold are selected for further analysis. Typical range: -0.5 to 0.0.
   * - ``cgcnn_batch_size``
     - integer
     - 256
     - Batch size for CGCNN inference. Larger values may improve throughput but require more GPU memory.

VASP Parameters
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 50

   * - Key
     - Type
     - Default
     - Description
   * - ``vasp_nnodes``
     - integer
     - 1
     - Number of GPU nodes allocated for VASP calculations. Each structure is calculated on this many nodes in parallel.
   * - ``vasp_ntasks_per_run``
     - integer
     - 1
     - Number of MPI processes per VASP calculation. Useful for CPU-only Parsl configurations where MPI parallelism is needed.
   * - ``vasp_nstructures``
     - integer
     - -1
     - Number of structures to process with VASP. ``-1`` means process all selected structures. Positive values limit the batch size for testing or resource constraints.
   * - ``vasp_timeout``
     - integer
     - 1800
     - Maximum walltime in seconds for each VASP calculation. Calculations exceeding this limit are terminated. Typical range: 1800-7200.
   * - ``vasp_nsw``
     - integer
     - 100
     - VASP ``NSW`` parameter: maximum number of ionic relaxation steps. See `VASP documentation <https://www.vasp.at/wiki/index.php/NSW>`_ for details.

Post-Processing Parameters
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 10 50

   * - Key
     - Type
     - Default
     - Description
   * - ``post_processing_output_dir``
     - string
     - ``""``
     - Absolute path to the directory that will contain post-processing results (convex hull plots, selected structures, etc.). If empty or omitted, the post-processing step is skipped.
   * - ``mp_rester_api_key``
     - string
     - ``""``
     - Materials Project API key for accessing reference stable phases. Obtain from https://docs.materialsproject.org. **Required** when ``post_processing_output_dir`` is set.
   * - ``hull_energy_threshold``
     - float
     - 0.1
     - Maximum E\ :sub:`hull` (eV/atom) threshold for displaying metastable phases in the convex hull visualization. Structures with E\ :sub:`hull` below this value are considered potentially synthesizable.

Workflow-Specific Parameters
=============================

MLIP Workflow
-------------

The following parameters are **required** for the ``mlip`` workflow:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Requirement
   * - ``post_processing_output_dir``
     - Must be specified (non-empty). Required for MLIP hull sorting stage.
   * - ``mp_rester_api_key``
     - Must be specified (non-empty). Required for accessing reference stable phases during hull sorting.
   * - ``mlip_relax_nnodes``
     - Should be set based on available GPU resources.
   * - ``gpus_per_node``
     - Should match the actual number of GPUs per node in your HPC system.

See :doc:`mlip_workflow` for a complete description of the MLIP workflow stages and configuration example.

Conditional Requirements
------------------------

.. important::

   **Materials Project API Key**: The ``mp_rester_api_key`` parameter becomes **required** when ``post_processing_output_dir`` is specified, regardless of which workflow is selected. Without this key, the post-processing stage cannot access reference stable phases for convex hull construction.

Command-Line Overrides
======================

All required and optional configuration parameters listed in this reference can be overridden via command-line arguments. The command-line value takes precedence over the JSON file value.

**Syntax:**

.. code-block:: bash

   exa_amd --config <json_file> --<parameter_name> <value>

**Examples:**

Override the number of workers:

.. code-block:: bash

   exa_amd --config configs/perlmutter.json --num_workers 256

Override multiple parameters:

.. code-block:: bash

   exa_amd --config configs/perlmutter.json --vasp_nnodes 4 --vasp_timeout 3600

View all available command-line options:

.. code-block:: bash

   exa_amd --help

Configuration Validation
=========================

exa-AMD performs automatic validation when loading configurations:

1. **Required parameter check**: All required parameters must be present
2. **Type coercion**: Command-line overrides are type-coerced based on the parameter definition
3. **Path creation**: Work directories are created automatically if they don't exist
4. **Element system validation**: Only ternary (3 elements) and quaternary (4 elements) systems are supported
5. **POTCAR generation**: POTCAR files are automatically generated from individual element potentials

Common Validation Errors
-------------------------

.. code-block:: text

   Error: Missing required argument 'workflow'

**Solution**: Add ``"workflow": "vasp"`` or ``"workflow": "mlip"`` to your JSON file.

**Missing Materials Project API Key**

**Condition**: When ``post_processing_output_dir`` is specified but ``mp_rester_api_key`` is not provided.

**Solution**: Provide your Materials Project API key when using post-processing. Either add ``"mp_rester_api_key": "your_key"`` to the JSON file or use ``--mp_rester_api_key your_key`` on the command line.

.. code-block:: text

   exa-AMD only supports ternary and quaternary systems

**Solution**: Ensure your ``elements`` parameter has exactly 3 or 4 elements separated by hyphens (e.g., ``"Ce-Co-B"`` or ``"Na-B-H-C"``).

Complete Configuration Examples
================================

Minimal VASP Workflow
---------------------

.. code-block:: json

   {
     "workflow": "vasp",
     "work_dir": "/path/to/work_dir",
     "vasp_work_dir": "/path/to/vasp_work_dir",
     "vasp_std_exe": "vasp_std",
     "vasp_pot_dir": "/path/to/potpaw_PBE",
     "vasp_output_file": "vasp_results.csv",
     "elements": "Ce-Co-B",
     "initial_structures_dir": "/path/to/initial_structures",
     "parsl_config": "perlmutter_premium",
     "parsl_configs_dir": "/path/to/parsl_configs"
   }

Full MLIP Workflow
------------------

.. code-block:: json

   {
     "workflow": "mlip",
     "work_dir": "/path/to/work_dir",
     "vasp_work_dir": "/path/to/vasp_work_dir",
     "vasp_std_exe": "vasp_std",
     "vasp_pot_dir": "/path/to/potpaw_PBE",
     "vasp_output_file": "vasp_results.csv",
     "elements": "Y-Mn-B",
     "initial_structures_dir": "/path/to/initial_structures",
     "parsl_config": "perlmutter_premium_mlip",
     "parsl_configs_dir": "/path/to/parsl_configs",
     "cpu_account": "m1234",
     "gpu_account": "m1234_g",
     "formation_energy_threshold": -0.2,
     "num_workers": 128,
     "cgcnn_batch_size": 256,
     "pre_processing_nnodes": 4,
     "mlip_relax_nnodes": 4,
     "gpus_per_node": 4,
     "vasp_nnodes": 1,
     "vasp_nstructures": 1000,
     "vasp_nsw": 100,
     "vasp_timeout": 1800,
     "hull_energy_threshold": 0.1,
     "post_processing_output_dir": "/path/to/post_processing_out_dir",
     "mp_rester_api_key": "your_mp_api_key_here"
   }

Quick Reference Table
=====================

All Parameters at a Glance
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Parameter
     - Type
     - Required
     - Default
   * - ``workflow``
     - string
     - Yes
     - —
   * - ``work_dir``
     - string
     - Yes
     - —
   * - ``vasp_work_dir``
     - string
     - Yes
     - —
   * - ``vasp_std_exe``
     - string
     - Yes
     - —
   * - ``vasp_pot_dir``
     - string
     - Yes
     - —
   * - ``vasp_output_file``
     - string
     - Yes
     - —
   * - ``elements``
     - string
     - Yes
     - —
   * - ``initial_structures_dir``
     - string
     - Yes
     - —
   * - ``parsl_config``
     - string
     - Yes
     - —
   * - ``parsl_configs_dir``
     - string
     - Yes
     - —
   * - ``formation_energy_threshold``
     - float
     - No
     - -0.2
   * - ``num_workers``
     - integer
     - No
     - 128
   * - ``cgcnn_batch_size``
     - integer
     - No
     - 256
   * - ``vasp_nnodes``
     - integer
     - No
     - 1
   * - ``vasp_ntasks_per_run``
     - integer
     - No
     - 1
   * - ``vasp_nstructures``
     - integer
     - No
     - -1
   * - ``vasp_timeout``
     - integer
     - No
     - 1800
   * - ``vasp_nsw``
     - integer
     - No
     - 100
   * - ``cpu_account``
     - string
     - No
     - ""
   * - ``gpu_account``
     - string
     - No
     - ""
   * - ``output_level``
     - string
     - No
     - "INFO"
   * - ``post_processing_output_dir``
     - string
     - Conditional*
     - ""
   * - ``mp_rester_api_key``
     - string
     - Conditional*
     - ""
   * - ``hull_energy_threshold``
     - float
     - No
     - 0.1
   * - ``pre_processing_nnodes``
     - integer
     - No
     - 1
   * - ``mlip_relax_nnodes``
     - integer
     - No
     - 1
   * - ``gpus_per_node``
     - integer
     - No
     - 4

\* ``post_processing_output_dir`` and ``mp_rester_api_key`` are required for MLIP workflow; ``mp_rester_api_key`` is required when ``post_processing_output_dir`` is set for any workflow.

See Also
========

- :doc:`quickstart` — Quick installation and basic usage
- :doc:`tutorial` — Step-by-step tutorial for running exa-AMD
- :doc:`workflow` — VASP workflow description
- :doc:`mlip_workflow` — MLIP workflow description
- :doc:`parsl_config` — Parsl configuration guide

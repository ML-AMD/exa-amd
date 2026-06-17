.. _mlip_workflow:

=============================================
Module 2: MLIP Relaxation and Hull Sorting
=============================================

The ``mlip`` workflow is Module 2 in *exa-AMD*. It extends the original
CGCNN-to-DFT workflow by inserting a machine-learning interatomic potential
(MLIP) relaxation stage and an MLIP-based hull-energy sorting stage before VASP
validation. The purpose is to spend expensive DFT calculations on structures
that have already been geometrically relaxed and ranked against the current
convex hull.

This module was added after the original exa-AMD workflow and is motivated by
the Y-Mn-B development case, where a large generated candidate pool required a
more selective bridge between fast formation-energy screening and first-principles
validation. In that workflow, CGCNN first reduced the generated structure set,
MLIP relaxation refined the selected structures, and hull sorting prioritized
low-energy candidates for DFT.

When to use this workflow
=========================

Use ``workflow: "mlip"`` when:

- the generated candidate pool is large enough that DFT validation of every
  CGCNN-selected structure is impractical;
- an MLIP model is suitable for the chemistry being screened;
- GPU resources are available for the MLIP relaxation stage; and
- convex-hull ranking is needed before selecting the final DFT queue.

For small candidate sets or systems where the MLIP model is not appropriate,
the standard :doc:`workflow` may be the safer choice.

Workflow stages
===============

The ``mlip`` workflow is registered as :class:`workflows.mlip_workflow.MLIPWorkflow`
and runs the following stages:

1. Structure generation
-----------------------

Candidate structures are generated from the configured initial structures and
target elements. This stage is shared with the standard VASP workflow.

**Main output**
  ``work_dir/<elements>/structures/``

2. CGCNN screening
------------------

CGCNN predicts formation energies for the generated candidates. The predictions
are collected in ``test_results.csv`` and used to select a larger intermediate
set for MLIP relaxation.

**Main output**
  ``work_dir/<elements>/test_results.csv``

3. Initial structure selection
------------------------------

The MLIP workflow writes the first selected structure pool to ``new0``. This
pool is intentionally larger than the final DFT queue so that MLIP relaxation
and hull sorting can make the next prioritization decision.

**Main output**
  ``work_dir/<elements>/new0/POSCAR_*``

4. MLIP relaxation
------------------

The selected ``new0/POSCAR_*`` files are relaxed using the FAIRChem UMA model
checkpoint expected at ``ml_models/mlip/uma-s-1p1.pt`` in a source checkout. The
relaxation helper uses ASE with a FIRE optimizer and writes one relaxed
structure and one energy record per candidate.

The model checkpoint is managed with Git LFS. Before cloning or running this
workflow, initialize LFS and make sure the checkpoint is present:

.. code-block:: bash

   git lfs install
   git lfs pull

**Main outputs**

- ``work_dir/<elements>/mlip_logs/CONTCAR_<id>``
- ``work_dir/<elements>/mlip_logs/energy_<id>.tmp``
- ``work_dir/<elements>/ener_ml.dat``

5. MLIP hull sorting
--------------------

The MLIP-relaxed energies are compared against the Materials Project-derived
stable-phase reference set used by the post-processing pipeline. The workflow
computes MLIP-based hull distances for ternary or quaternary systems and sorts
the candidates from lowest to highest hull energy.

This stage requires both:

- ``post_processing_output_dir``
- ``mp_rester_api_key``

**Main output**
  ``work_dir/<elements>/hull_ml.dat``

6. DFT validation queue
-----------------------

The sorted MLIP candidates are copied from ``mlip_logs/CONTCAR_<id>`` into the
standard DFT input directory as ``new/POSCAR_<rank>``. VASP then validates the
top-ranked candidates according to ``vasp_nstructures``. Set
``vasp_nstructures`` to ``-1`` to process all remaining structures.

**Main outputs**

- ``work_dir/<elements>/new/POSCAR_*``
- ``vasp_work_dir/<elements>/<id>/``
- ``vasp_work_dir/<elements>/vasp_results.csv``

7. Final post-processing
------------------------

The validated VASP results are post-processed with the standard exa-AMD convex
hull workflow. This produces the final hull plot and the selected stable or
metastable candidates.

**Main outputs**

- ``post_processing_output_dir/hull_plot.png``
- ``post_processing_output_dir/selected/``

Configuration
=============

Set the workflow and Parsl configuration to the MLIP-enabled entries:

.. code-block:: json

   {
     "workflow": "mlip",
     "parsl_config": "perlmutter_premium_mlip",
     "parsl_configs_dir": "<abs_path_to>/parsl_configs",
     "cpu_account": "<cpu_account>",
     "gpu_account": "<gpu_account>",
     "elements": "Y-Mn-B",
     "work_dir": "<abs_path_to>/work_dir",
     "vasp_work_dir": "<abs_path_to>/vasp_work_dir",
     "initial_structures_dir": "<abs_path_to>/initial_structures",
     "vasp_pot_dir": "<abs_path_to>/potpaw_PBE",
     "formation_energy_threshold": -0.2,
     "num_workers": 128,
     "cgcnn_batch_size": 256,
     "vasp_std_exe": "vasp_std",
     "vasp_output_file": "vasp_results.csv",
     "pre_processing_nnodes": 4,
     "mlip_relax_nnodes": 4,
     "gpus_per_node": 4,
     "vasp_nnodes": 1,
     "vasp_nstructures": 1000,
     "vasp_nsw": 100,
     "vasp_timeout": 1800,
     "hull_energy_threshold": 0.1,
     "post_processing_output_dir": "<abs_path_to>/post_processing_out_dir",
     "mp_rester_api_key": "<MP_RESTER_API_KEY>"
   }

The most important MLIP-specific keys are:

.. list-table:: MLIP workflow keys
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Purpose
   * - ``workflow``
     - Use ``"mlip"`` to select :class:`workflows.mlip_workflow.MLIPWorkflow`.
   * - ``parsl_config``
     - Use an MLIP-capable Parsl config, such as ``"perlmutter_premium_mlip"``.
   * - ``mlip_relax_nnodes``
     - Number of GPU nodes allocated to MLIP relaxation.
   * - ``gpus_per_node``
     - Number of GPUs per node used to partition MLIP relaxation work.
   * - ``post_processing_output_dir``
     - Directory used to build reference hull data and final hull outputs.
   * - ``mp_rester_api_key``
     - Materials Project API key required for reference stable phases.
   * - ``vasp_nstructures``
     - Number of hull-sorted MLIP candidates to validate with VASP.

Run command
===========

From the repository root:

.. code-block:: bash

   exa_amd --config configs/my_mlip_config.json

or, when running from source:

.. code-block:: bash

   python exa_amd.py --config configs/my_mlip_config.json

Output layout
=============

For an ``elements`` value such as ``Y-Mn-B``, the main intermediate files are:

.. code-block:: text

   work_dir/
   `-- Y-Mn-B/
       |-- structures/
       |-- test_results.csv
       |-- new0/
       |   `-- POSCAR_*
       |-- mlip_logs/
       |   |-- CONTCAR_*
       |   `-- energy_*.tmp
       |-- ener_ml.dat
       |-- hull_ml.dat
       `-- new/
           `-- POSCAR_*

   vasp_work_dir/
   `-- Y-Mn-B/
       |-- mp_int_stable.dat
       |-- energy.dat
       |-- vasp_results.csv
       `-- <vasp job id>/

   post_processing_output_dir/
   |-- hull_plot.png
   `-- selected/

Troubleshooting
===============

Missing UMA checkpoint
  If ``ml_models/mlip/uma-s-1p1.pt`` is a small pointer file or missing, install
  Git LFS and run ``git lfs pull`` from the repository root.

Missing Materials Project key
  The MLIP workflow requires ``post_processing_output_dir`` and
  ``mp_rester_api_key`` because hull sorting needs reference stable phases before
  DFT validation.

No ``new/POSCAR_*`` files after MLIP sorting
  Check that ``hull_ml.dat`` exists and that each listed candidate has a matching
  ``mlip_logs/CONTCAR_<id>`` file.

Unexpected DFT queue size
  Check ``vasp_nstructures``. A positive value limits the next VASP batch; ``-1``
  processes all remaining structures.

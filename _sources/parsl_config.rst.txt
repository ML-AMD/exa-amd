.. _parsl_config:

Parsl configuration guide
=========================

.. note::
   This guide focuses on **Slurm-based** configurations for brevity. Parsl also
   supports a wide range of resource providers and launchers beyond Slurm, and
   can run on many different systems. See the Parsl documentation for other
   workload managers.

exa-AMD uses `Parsl <https://parsl-project.org>`__ to schedule each workflow
phase on your system. exa-AMD typically uses Parsl *configuration* to specify Slurm accounts/queues,
CPU/GPU placement, number of nodes, per-node workers/threads, walltime, and how
the software environment is initialized on compute nodes.

Parsl can technically run the workflow on a local machine. However, exa-AMD is designed for supercomputers and typically requires substantial computational resources, so local execution is not recommended.


Executor labels
---------------

exa-AMD uses labels to link tasks to executors. A task’s label selects the executor it will run on.
The executor, in turn, defines the computational resources used by the task (e.g., CPUs/GPs, node count).


First, the label is used when registering the executor:

.. code-block:: python

   executor = HighThroughputExecutor(
       label=EXECUTOR_LABEL,
       ...
   )

Then, the same label is referenced by the task decorator to run that task on that executor:

.. code-block:: python

   @python_app(executors=[EXECUTOR_LABEL])
   def task():


This keeps the workflow code independent of site details while precisely controlling resource placement per task.

exa-AMD uses fixed executor labels for each of the five workflow phases described in :doc:workflow:

.. list-table:: Executor labels
   :header-rows: 1
   :widths: 35 65

   * - Phase
     - Parsl executor label
   * - Structure generation
     - ``GENERATE_EXECUTOR_LABEL``
   * - CGCNN prediction
     - ``CGCNN_EXECUTOR_LABEL``
   * - Structure selection
     - ``SELECT_EXECUTOR_LABEL``
   * - VASP (DFT)
     - ``VASP_EXECUTOR_LABEL``
   * - Post-processing
     - ``POSTPROCESSING_LABEL``


Selecting a config at runtime
-----------------------------

At runtime, exa-AMD reads a **JSON** run configuration. The key:

- ``parsl_config`` — selects which **registered Parsl config** to use
  (e.g., ``"perlmutter_premium"``).

This value must **match the registry name** defined in the Python config, e.g.:

.. code-block:: python

   # https://github.com/ML-AMD/exa-amd/blob/main/parsl_configs/perlmutter.py#L154
   register_parsl_config("perlmutter_premium", PerlmutterConfig)

Built-in configurations
-----------------------

The repository provides four registered configurations:

.. code-block:: python

   register_parsl_config("chicoma", ChicomaConfig)
   register_parsl_config("chicoma_debug", ChicomaConfigDebug)
   register_parsl_config("chicoma_debug_cpu", ChicomaConfigDebugCPU)
   register_parsl_config("perlmutter_premium", PerlmutterConfig)

Select any of these by setting ``parsl_config`` accordingly in your run JSON.

Using the provided configs
--------------------------

For **LANL Chicoma** and **NERSC Perlmutter**:

- Update **``worker_init``** in the config to load your site modules and activate
  your Conda environment *on compute nodes*.
- Provide **accounts** and other runtime knobs (e.g., number of nodes) in your run JSON.


Additionally, update the different Slurm fields (e.g., ``qos``, ``constraint``,
launcher).

Registering a new config
------------------------

If your site differs substantially you may want to register a new Parsl configuration:

1. Create a file under ``parsl_configs/`` (e.g., ``my_system.py``).
2. Implement a ``Config`` subclass with five executors (using the labels above).
3. **Register** it with a **unique** name:

   .. code-block:: python

      register_parsl_config("my_system", MySystemConfig)

4. Set ``parsl_config`` to ``"my_system"`` in your run JSON.

.. important::
   The registry name must be **unique** across all registered configs in your
   environment.

Resource allocation & placement
-------------------------------

Parsl’s provider/executor fields map directly to the resources you request from Slurm.

**Node type**
  - ``constraint``: choose CPU vs GPU nodes (e.g., ``"cpu"`` or ``"gpu"``).
  - ``available_accelerators``: GPUs *per node* (e.g., 4 on Perlmutter).

**How many nodes**
  - ``nodes_per_block``: nodes in one Slurm allocation.
  - ``max_blocks`` / ``min_blocks`` / ``init_blocks``: how many allocations Parsl may keep alive.
    - One multi-node allocation: ``nodes_per_block = N``, ``max_blocks = 1``.
    - Many single-node allocations: ``nodes_per_block = 1``, ``max_blocks = N``.

**Per-node concurrency**
  - ``cores_per_worker``: CPUs per Parsl worker.
  - ``max_workers_per_node``: limit on workers per node.

**Operational**
  - ``account`` and ``qos``: indetical to Slurm equivalents.
  - ``walltime``: job time limit.
  - ``worker_init``: environment on compute nodes (e.g., modules).
  - ``scheduler_options``: raw ``#SBATCH`` directives when needed.

Quick mapping to Slurm
~~~~~~~~~~~~~~~~~~~~~~

- Nodes: ``nodes_per_block`` → roughly ``-N``.
- GPUs per node: ``available_accelerators`` → akin to ``--gpus-per-node``.
- CPU threads per worker: ``cores_per_worker`` → similar to ``--cpus-per-task`` (per worker).
- Multiple allocations: ``max_blocks`` > 1 → multiple Slurm jobs managed by Parsl.

What the run JSON typically controls
------------------------------------

Common knobs provided in the run JSON (names may vary slightly by version):

- **Parsl selection & accounts**
  - ``parsl_config``: registry name of the site config (e.g., ``"perlmutter_premium"``).
  - ``cpu_account`` / ``gpu_account``: Slurm accounts for CPU/GPU executors.

- **Resource allocation & placement**
  - ``num_workers``: CPU threads per worker (used by CPU-bound phases).
  - ``pre_processing_nnodes``: node count for structure generation and CGCNN.
  - ``vasp_nnodes``: node count for the VASP phase.


Full working example
--------------------

For a complete configuration with five labeled executors and typical Slurm settings,
see the Perlmutter config in the repository:

- ``parsl_configs/perlmutter.py``

Need help?
----------

If you are setting up a new site configuration or encountering center-specific
constraints, please open a **discussion** or **issue**:

- https://github.com/ML-AMD/exa-amd

Further reading
---------------

- Parsl configuration guide:
  https://parsl.readthedocs.io/en/latest

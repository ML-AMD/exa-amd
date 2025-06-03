Tutorial
========

This tutorial walks through how to set up and run exa-AMD on [NERSC’s Perlmutter supercomputer](https://docs.nersc.gov/systems/perlmutter/architecture/), on the **GPU partition**. 

1. Clone the Repository
------------------------

Start by cloning the `exa-AMD` repository:

.. code-block:: bash

   git clone https://github.com/ML-AMD/exa-amd.git
   cd exa-amd

----

2. Install Dependencies
------------------------

Follow the :ref:`installation` instructions to create the required environment using Conda. After installation, make sure to activate the environment:

.. code-block:: bash

   conda activate amd_env

----

3. Prepare the Data and VASP Setup
-----------------------------------

Ensure you have a working `VASP <https://www.vasp.at>`_ installation and access to the latest PAW potentials containing kinetic energy densities for meta-GGA calculations .

If you do not already have it:

- Log into your account on the `VASP website <https://www.vasp.at>`_
- Navigate to the **Downloads** section
- Download the latest archive (e.g.,``potpaw_PBE.52.tar.gz``)

Once downloaded, extract the archive and place the resulting `potpaw_PBE` directory anywhere on your system. You will reference its location in your JSON config using the ``pot_dir`` parameter.

Next, `download <https://iastate.box.com/s/3swro78kbcd69fwamhk7df4n5rx4edae>`_ and extract the **`initial_structures`** dataset which contains a set of initial crystal structures used by the workflow.

4. Prepare the JSON Configuration File
---------------------------------------

Copy the default Perlmutter configuration:

.. code-block:: bash

   cp configs/perlmutter.json configs/my_config_perlmutter.json

Edit the following fields in `my_config_perlmutter.json`:

- `"cms_dir"`: Absolute path to your `cms_dir` directory
- `"work_dir"`: A scratch directory for intermediate files
- `"vasp_work_dir"`: A work directory for running VASP calculations
- `"pot_dir"`: Path to your `potpaw_PBE` directory
- `"initial_structures"`: Path to the `initial_structures` directory (downloaded in the previous section)

----

5. Prepare the Parsl Configuration
-----------------------------------

Parsl configurations must be placed inside the ``parsl_configs/`` directory so that they can be automatically discovered by exa-AMD at runtime.

Start by copying the default Perlmutter configuration:

.. code-block:: bash

   cp parsl_configs/perlmutter.py parsl_configs/my_perlmutter.py

Then edit `my_perlmutter.py`:

a. Change the registration name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the bottom of the file, update `register_parsl_config()` to reflect the new config name. This value **have to match** the value you will set in your JSON config file (the `"parsl_config"` field).

.. code-block:: python

   # Before:
   register_parsl_config("perlmutter_premium", PerlmutterConfig)

   # After:
   register_parsl_config("my_perlmutter", PerlmutterConfig)

b. Update each executor
~~~~~~~~~~~~~~~~~~~~~~~

The Perlmutter configuration defines **four separate executors**:

- Two that run on **GPU nodes** (for VASP and CGCNN tasks)
- Two that run on **CPU nodes** (for structure generation and selection)

For each executor, update the following fields in the `SlurmProvider`:

- `account`: your NERSC allocation account (e.g., `"m1234"`)
- `qos`: the QOS for that job (e.g., `"regular"`, `"premium"`)

.. code-block:: text

   The account and qos values used in the Parsl configuration are exactly the same
   as the ones you would provide when running with Slurm directly on Perlmutter,
   using commands like salloc, srun, or sbatch.

   For example, if you normally run:
     salloc -A m1234 -q regular -C gpu

   Then in your Parsl config, you should use:
     account="m1234"
     qos="regular"
     constraint="gpu"

Here is an example:

.. code-block:: python

   provider=SlurmProvider(
       account="your_gpu_account",    # ← CHANGE IF NEEDED
       qos="your_gpu_qos",            # ← CHANGE IF NEEDED
       constraint="gpu",
       ...
   )

.. note::

   The account can also be specified at runtime via the command-line arguments.

Make sure you update **all four** executors accordingly, using your appropriate account and qos for CPU and GPU resources.

.. important::

   All Parsl configuration files **must be placed inside the** ``parsl_configs/`` **directory**.


For more information about possible Parsl configurations, see the official documentation [#parsl_docs]_.

.. [#parsl_docs] https://parsl-project.org

c. Update JSON Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After registering the new Parsl configuration, update your JSON config file to reference it:

.. code-block:: text

   {
        ...
       "parsl_config": "my_perlmutter"
   }

exa-AMD will now automatically discover and use the `my_perlmutter` configuration at runtime.

----

6. Run the Workflow
---------------------

Once everything is configured, run the full exa-AMD workflow from a logind node of Perlmutter:

.. code-block:: bash

   export PYTHONPATH=$(pwd):$PYTHONPATH
   python amd.py --config configs/my_config_perlmutter.json --vasp_nnodes 2

This will launch the four steps:

1. :func:`~parsl_tasks.gen_structures.generate_structures` — structure generation
2. :func:`~parsl_tasks.cgcnn.run_cgcnn` — CGCNN prediction
3. :func:`~parsl_tasks.cgcnn.select_structures` — structure selecton
4. :func:`~parsl_tasks.vasp.vasp_calculations` — VASP relaxation and energy calculations

Progress and logs will be printed to stdout/stderr.

----

7. Check the Results
---------------------

After the workflow completes, you should verify that all stages ran successfully by inspecting
the contents of the work directory (`work_dir`) and the VASP work directory (`vasp_work_dir`).

a. Work directory
~~~~~~~~~~~~~~~~~

Inside your specified `work_dir`, you should see a subdirectory named after the elements string (i.e., `Na-B-C`) with the following contents:

.. code-block:: text

   work_dir/
   └── Na-B-C
       ├── new/ 
       ├── POTCAR 
       ├── structures/ 
       └── test_results.csv

b. VASP Directory
~~~~~~~~~~~~~~~~~~

Your `vasp_work_dir` will contain a subdirectory for each selected structure ID, where VASP calculations were run:

.. code-block:: text

   vasp_work_dir/
   └── Na-B-C
       ├── 1/
       ├── 2/
       ├── 3/
       ├── ...
       ├── 10/
       └── vasp_calc_result.csv  ← Final results summary

Each numbered folder corresponds to a VASP calculation for a selected structure.

c. Final Output
~~~~~~~~~~~~~~~

This file summarizes the outcome of each VASP calculation. A fully successful run should look like this:

.. code-block:: text

   id,result
   1,success
   2,success
   3,success
   4,success
   5,success
   6,success
   7,success
   8,success
   9,success
   10,success

If all lines show `success`, then the workflow completed as expected.

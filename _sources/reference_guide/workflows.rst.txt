Workflow steps
==============

- The steps use filesystem artifacts to determine completion.
- Steps are fail-fast: they terminate execution on critical failures.
- The same step implementation may be reused by multiple workflows.

Base interface
--------------

.. autoclass:: workflows.steps.Step
   :members:
   :show-inheritance:

Steps
-----

.. autoclass:: workflows.steps.GenerateStructuresStep
   :members:

.. autoclass:: workflows.steps.CgcnnStep
   :members:

.. autoclass:: workflows.steps.SelectStructuresStep
   :members:

.. autoclass:: workflows.steps.MLIPRelaxationStep
   :members:

.. autoclass:: workflows.steps.EhullMLParallel
   :members:

.. autoclass:: workflows.steps.VaspCalculationsStep
   :members:

.. autoclass:: workflows.steps.PostProcessingStep
   :members:


Workflows
=========

MLIP based
----------

1. :class:`workflows.steps.GenerateStructuresStep`
2. :class:`workflows.steps.CgcnnStep`
3. :class:`workflows.steps.SelectStructuresStep`
4. :class:`workflows.steps.MLIPRelaxationStep`
5. :class:`workflows.steps.EhullMLParallel`
6. :class:`workflows.steps.VaspCalculationsStep`
7. :class:`workflows.steps.PostProcessingStep`

VASP based
----------

1. :class:`workflows.steps.GenerateStructuresStep`
2. :class:`workflows.steps.CgcnnStep`
3. :class:`workflows.steps.SelectStructuresStep`
4. :class:`workflows.steps.VaspCalculationsStep`
5. :class:`workflows.steps.PostProcessingStep`
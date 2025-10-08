Workflow stages
===============

1. Structure generation
-----------------------

**Parsl executor:** ``GENERATE_EXECUTOR_LABEL``

Start from structures in the ``initial_structures`` directory and generate hypothetical structures:

- Perform elemental substitution: the elements in each initial structure are replaced with the target elements under investigation.
- Cover atomic arrangements by enumerating (or randomly shuffling) the order of substituted elements (i.e., all permutations for the given system).
- Apply lattice scaling (typically from 0.94 to 1.06) to span realistic bond-length variations, since optimal bond lengths for the new elements may differ from the original structure.
- The cross-product of element orderings and scale factors yields many variants:
  - Ternary: 30 variants per initial structure.
  - Quaternary: 24 possible orderings.

2. CGCNN-based structure screening
----------------------------------

**Parsl executor:** ``CGCNN_EXECUTOR_LABEL``

- Evaluate all generated structures with a Crystal Graph Convolutional Neural Network (CGCNN) to predict formation energies efficiently.
- Select structures with low predicted formation energy as promising candidates, reducing the cost of subsequent first-principles calculations.

3. Removal of similar structures
--------------------------------

**Parsl executor:** ``SELECT_EXECUTOR_LABEL``


- Identify and remove duplicates or near-duplicates using a structural-similarity threshold.
- The deduplication step ensures that only non-equivalent structures are retained, typically narrowing the set to a manageable number (e.g., 1,000–5,000 structures) for detailed study.

4. DFT calculations (relaxation and energy)
-------------------------------------------

**Parsl executor:** ``VASP_EXECUTOR_LABEL``

- The filtered set of structures is subjected to first-principles calculations using Density Functional Theory (DFT) using VASP (extensible to other ab initio codes such as Quantum ESPRESSO).
- Each structure undergoes full relaxation to find its lowest-energy geometry, followed by a self-consistent total-energy calculation.
- The resulting relaxed structures and total energies provide the basis for thermodynamic analysis.

5. Post-processing: convex hull and stability analysis
------------------------------------------------------

**Parsl executor:** ``POSTPROCESSING_LABEL``

- Determine the formation energies of each structure relative to known stable phases.
- Construct the convex hull to indentify structures that are:
  - Thermodynamically stable: on (or below) the current convex hull.
  - Metastable: low formation energy (< 0.05 eV/atom ) above the hull (Ehull < 0.05 eV/atom).
- This analysis reveals new stable and metastable structures and updates the phase diagram for the target system.

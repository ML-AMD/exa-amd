import os
from multiprocessing import Pool, cpu_count
from pymatgen.core import Composition, Element
from parsl import python_app

from tools.logging_config import amd_logger
from parsl_configs.parsl_executors_labels import EHULL_ML_PARALLEL_EXECUTOR_LABEL
from tools.config_labels import ConfigKeys as CK
from parsl_tasks.ehull_utils import (
    judge_stable_ternary,
    parse_stable_phases_ternary,
    judge_stable_quaternary,
    parse_stable_phases_quaternary,
)


def process_structure_wrapper_ternary(index, formula, energy, stable_vec, elements):
    """
    Wrapper function to be called by the worker pool.
    Returns the index, the result, and the formula.
    """
    try:
        d_hull, hull_vec = judge_stable_ternary(stable_vec, elements, formula, energy)
        return index, d_hull, hull_vec, formula
    except Exception as e:
        # Return None for energy if calculation fails
        return index, None, None, formula


def process_structure_wrapper_quaternary(index, formula, energy, stable_vec, elements):
    """
    Wrapper function to be called by the worker pool.
    """
    try:
        d_hull, hull_vec = judge_stable_quaternary(stable_vec, elements, formula, energy)
        return index, d_hull, hull_vec, formula
    except Exception as e:
        # Return None for energy if calculation fails
        return index, None, None, formula


def read_energies(filename):
    energies = []
    formulas = []
    indices = []
    with open(filename, 'r') as f:
        for line in f:
            l_parts = line.split(',')
            index = int(l_parts[0])
            indices.append(index)
            energies.append(float(l_parts[1]))
            formulas.append(l_parts[2].strip())
    return energies, indices, formulas


def parallel_ternary_ehull(input_file,
                           stable_file,
                           output_file,
                           elements,
                           workers=None):
    """
    Calculate formation energies relative to the ternary convex hull (parallel).

    Parameters
    ----------
    input_file : str
        Input energy file with lines: index,energy,formula. (e.g., ener_ml.dat)
    stable_file : str
        Stable phases file with lines: formula energy. (e.g., mp_int_stable.dat)
    output_file : str
        Output file written as: index,Ehull. (e.g., hull_ml.dat)
    elements : str or list, optional
        Element specification (e.g. "A-B-C"). If None, inferred from current directory.
    workers : int, optional
        Number of worker processes to use. Defaults to all available CPUs.

    Returns
    -------
    str
        Path to the output file.
    """
    if workers is None or workers < 1:
        workers = cpu_count()

    elements = [Element(ele) for ele in elements.split('-')]
    eles = [ele.symbol for ele in elements]

    # Read stable phases
    stable_vec, ternary_vec = parse_stable_phases_ternary(stable_file, elements)
    if not stable_vec:
        amd_logger.critical(f"Error: No stable phases found containing elements {eles}")

    for ternaries in ternary_vec:
        amd_logger.debug(f"{ternaries[0]}")

    # Read Input Structures
    total_energies, indices, formulas = read_energies(input_file)
    amd_logger.debug(f"\nStarting parallel calculation on {len(indices)} structures using {workers} workers...")

    # Prepare arguments for each task
    task_args = [
        (idx, form, en, stable_vec, eles)
        for idx, form, en in zip(indices, formulas, total_energies)
    ]

    # Create a Pool and run tasks
    with Pool(processes=workers) as pool:
        results = pool.starmap(process_structure_wrapper_ternary, task_args)

    formation_energies = []
    hull_phases = []
    processed_indices = []
    processed_formulas = []

    for idx, d_hull, hull_vec, form in results:
        processed_indices.append(idx)
        processed_formulas.append(form)
        formation_energies.append(d_hull)
        hull_phases.append(hull_vec)

        if d_hull is None:
            amd_logger.warning(f"Warning: Calculation failed for structure {idx}")

    # Sort results based on formation energy (Low to High)
    # Filter out None values before sorting to avoid crashes
    valid_data = [
        (e, f, i, h)
        for e, f, i, h in zip(formation_energies, processed_formulas, processed_indices, hull_phases)
        if e is not None
    ]

    if not valid_data:
        amd_logger.critical("No valid calculations found.")

    # Sort based on energy (first element of tuple)
    valid_data.sort(key=lambda x: x[0])

    # Unpack sorted data for writing
    sorted_energies, sorted_formulas, sorted_indices, sorted_phases = zip(*valid_data)

    # Write output
    with open(output_file, 'w+') as f:
        for idx, energy, phases, formula in zip(sorted_indices, sorted_energies, sorted_phases, sorted_formulas):
            f.write(f'{idx},{energy:.6f}\n')
    return output_file


def parallel_quaternary_ehull(input_file,
                              stable_file,
                              output_file,
                              elements,
                              workers=None):
    if workers is None or workers < 1:
        workers = cpu_count()

    elements = [Element(ele) for ele in elements.split('-')]
    eles = [ele.symbol for ele in elements]

    if len(eles) != 4:
        amd_logger.critical(f"Error: Detected {len(eles)} elements ({eles}). This function is for Quaternary (4) systems only.")

    stable_vec, _ = parse_stable_phases_quaternary(stable_file, elements)
    if not stable_vec:
        amd_logger.critical(f"Error: No stable phases found for system {'-'.join(eles)}")

    total_energies, indices, formulas = read_energies(input_file)
    amd_logger.debug(f"\nStarting parallel calculation on {len(indices)} structures using {workers} workers...")

    task_args = [
        (idx, form, en, stable_vec, eles)
        for idx, form, en in zip(indices, formulas, total_energies)
    ]

    with Pool(processes=workers) as pool:
        results = pool.starmap(process_structure_wrapper_quaternary, task_args)

    formation_energies = []
    hull_phases = []
    processed_indices = []
    processed_formulas = []

    for idx, d_hull, hull_vec, form in results:
        # Filter out failed calculations
        if d_hull is not None and d_hull > -99.0:  # Check for validity
            processed_indices.append(idx)
            processed_formulas.append(form)
            formation_energies.append(d_hull)
            hull_phases.append(hull_vec)
        elif d_hull is None:
            amd_logger.warning(f"Warning: Calculation failed for structure {idx}")

    valid_data = list(zip(formation_energies, processed_formulas, processed_indices, hull_phases))

    if not valid_data:
        amd_logger.critical("No valid calculations found.")

    valid_data.sort(key=lambda x: x[0])

    sorted_energies, sorted_formulas, sorted_indices, sorted_phases = zip(*valid_data)

    with open(output_file, 'w+') as f:
        for idx, energy, phases, formula in zip(sorted_indices, sorted_energies, sorted_phases, sorted_formulas):
            f.write(f'{idx},{energy:.6f}\n')

    return output_file


@python_app(executors=[EHULL_ML_PARALLEL_EXECUTOR_LABEL])
def ehull_ml_parallel(config):
    ener_ml_file = os.path.join(config[CK.WORK_DIR], CK.MLIP_ENER_ML_FILE)
    mp_file = os.path.join(config[CK.VASP_WORK_DIR], CK.MP_STABLE_OUT)
    output_file = os.path.join(config[CK.WORK_DIR], CK.MLIP_HULL_ML_FILE)
    elements = config[CK.ELEMENTS]

    n = len(elements.split('-'))
    if n == 3:
        return parallel_ternary_ehull(ener_ml_file, mp_file, output_file, elements)
    if n == 4:
        amd_logger.info("ehull ")
        return parallel_quaternary_ehull(ener_ml_file, mp_file, output_file, elements)

    amd_logger.critical(f"Unsupported number of elements ({n}) for system='{elements}'")

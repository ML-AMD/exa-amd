import sys
import os
import re
import warnings
from ase.io import read, write
from ase.optimize import LBFGS, FIRE
from ase.filters import FrechetCellFilter
# Note: fairchem libraries must be importable in each worker process
from fairchem.core import pretrained_mlip, FAIRChemCalculator

# Global variables for model/calculator storage (set in initializer)
global_predictor = None
global_calc = None
global_model_path = None


def worker_initializer(model_path_val):
    """
    Called once when each worker process starts. Loads the model into memory.
    """
    global global_predictor, global_calc, global_model_path

    # CRITICAL: Set the visible device for this specific process
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', 'Unknown')

    print(f"Worker on GPU {gpu_id}: Initializing model from {model_path_val}...")
    try:
        # 1. Load the model once
        global_predictor = pretrained_mlip.load_predict_unit(model_path_val, device="cuda")
        global_calc = FAIRChemCalculator(global_predictor, task_name="omat")
        global_model_path = model_path_val
        print(f"Worker on GPU {gpu_id}: Initialization complete.")
    except Exception as e:
        print(f"Worker on GPU {gpu_id}: Error during model loading: {e}", file=sys.stderr)
        sys.exit(1)


def relax_and_log(input_cif, energy_log_dir):
    """
    The main relaxation function called by the pool worker.
    The model (global_calc) is already loaded.
    """
    # Use the global calculator loaded by the initializer
    calc = global_calc

    # 1. Determine base name and index
    base_name = os.path.splitext(os.path.basename(input_cif))[0]
    match = re.search(r'\d+', base_name)
    index = match.group(0) if match else base_name

    out_vasp_name = os.path.join(energy_log_dir, f"CONTCAR_{index}")
    temp_log_path = os.path.join(energy_log_dir, f"energy_{index}.tmp")

    try:
        # 2. Setup ASE atoms
        atoms = read(input_cif)
        atoms.calc = calc

        # 3. Perform Relaxation
        # opt = LBFGS(FrechetCellFilter(atoms))
        opt = FIRE(FrechetCellFilter(atoms))
        opt.run(fmax=0.05, steps=100)

        # 4. Write relaxed structure and get energy
        write(out_vasp_name, atoms, format='vasp')
        energy = atoms.get_potential_energy()
        num_atoms = len(atoms)
        energy_per_atom = energy / num_atoms
        formula = atoms.symbols.get_chemical_formula()

        # 5. Write results to unique temporary file
        with open(temp_log_path, 'w') as f:
            f.write(f"{index},{energy_per_atom},{formula}\n")

        return f"Successfully relaxed {input_cif} ({energy_per_atom:.4f} eV/atom)"

    except FileNotFoundError as e:
        error_msg = f"Error relaxing {input_cif}: output directory not found ({e})"
        print(error_msg, file=sys.stderr)
        return error_msg

    except Exception as e:
        error_msg = f"Error relaxing {input_cif}: {e}"
        print(error_msg, file=sys.stderr)
        with open(temp_log_path, 'w') as f:
            f.write(f"{index},ERROR\n")
        return error_msg


def main():
    # New check: needs at least 3 arguments (model, log_dir, file1)
    if len(sys.argv) < 4:
        print("Usage: python uma_pool.py <model_path> <energy_log_dir> <input_file1> ...")
        sys.exit(1)

    model_path = sys.argv[1]
    energy_log_dir = sys.argv[2]
    input_files = sys.argv[3:]

    warnings.filterwarnings("ignore")

    worker_initializer(model_path)
    for file in input_files:
        relax_and_log(file, energy_log_dir)


if __name__ == "__main__":
    main()

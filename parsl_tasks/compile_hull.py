from parsl import bash_app, python_app
from tools.config_labels import ConfigKeys as CK
from parsl_configs.parsl_executors_labels import POSTPROCESSING_LABEL


def cmd_compile_vasp_hull(config, total_calcs, output_file, prefix):
    '''
    collect convex hull for all the structures
    '''
    import os

    total_calcs = int(total_calcs)
    pairs = []

    for calc_idx in range(1, total_calcs + 1):
        calc_dir = f"{prefix}{calc_idx}"
        incar = os.path.join(calc_dir, "INCAR")
        out = os.path.join(calc_dir, "output")
        outcar = os.path.join(calc_dir, "OUTCAR")

        formula = ""
        energy = None
        natoms = None

        try:
            last = None
            with open(incar, "r") as f:
                for ln in f:
                    if "SYSTEM" in ln:
                        last = ln
            if last:
                toks = last.split()
                if len(toks) >= 3:
                    formula = toks[2]
        except FileNotFoundError:
            pass

        try:
            last = None
            with open(out, "r") as f:
                for ln in f:
                    if "F=" in ln:
                        last = ln
            if last:
                toks = last.split()
                if len(toks) >= 5:
                    energy = float(toks[4])
        except FileNotFoundError:
            pass

        try:
            last = None
            with open(outcar, "r") as f:
                for ln in f:
                    if "NIONS" in ln:
                        last = ln
            if last:
                natoms = int(last.split()[-1])
        except FileNotFoundError:
            pass

        if (energy is not None) and (natoms is not None):
            tenergy = float(f"{energy:.6f}")
            epa = float(f"{tenergy / natoms:.6f}")
            pairs.append((formula, epa))

    pairs.sort(key=lambda x: (x[0], x[1]))
    seen = set()
    with open(output_file, "w") as f:
        for formula, epa in pairs:
            if formula not in seen:
                seen.add(formula)
                f.write(f"{formula} {epa:.6f}\n")


@python_app(executors=[POSTPROCESSING_LABEL])
def compile_vasp_hull(config, total_calcs, output_file, prefix):
    return cmd_compile_vasp_hull(config, total_calcs, output_file, prefix)

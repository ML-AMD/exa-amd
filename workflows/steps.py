import os
import time
import pandas as pd
import glob
import parsl
import math
import sys

from pathlib import Path
from abc import ABC, abstractmethod

from tools.errors import VaspNonReached
from parsl.app.errors import AppTimeout
from parsl.app.errors import BashExitFailure
from tools.logging_config import amd_logger
from tools.config_manager import ConfigManager
from tools.config_labels import ConfigKeys as CK
from parsl_configs.parsl_executors_labels import *

from parsl_tasks.ehull import calculate_ehul
from parsl_tasks.convex_hull import convex_hull_color
from parsl_tasks.ml_ehull import ehull_ml_parallel
from tools.post_processing import get_vasp_hull

STATUS_BY_EXCEPTION = {
    VaspNonReached: "non_reached",
    AppTimeout: "time_out",
    BashExitFailure: "bash_exit_failure",
}


def _write_status(fp, id_, status):
    fp.write(f"{id_},{status}\n")


def _collect_future_errors(futures, step_name: str) -> bool:
    errors = []
    for i, f in enumerate(futures, start=1):
        exc = f.exception()
        if exc is not None:
            errors.append((i, exc))
    if errors:
        msgs = "; ".join(f"#{i}: {type(e).__name__}: {e}" for i, e in errors)
        amd_logger.critical(f"{step_name} task failures: {msgs}")


class Step(ABC):
    """Minimal step interface: not_finished + run, bound to a config."""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def not_finished(self) -> bool:
        ...

    @abstractmethod
    def run(self) -> None:
        ...


class GenerateStructuresStep(Step):

    def _generate_structures(self):
        """
        Generate hypothetical structures in parallel (chunked).

        Submits :func:`parsl_tasks.gen_structures.gen_structures` for each chunk
        ``i ∈ [1, n_chunks]`` where ``n_chunks = config[CK.GEN_STRUCTURES_NNODES]``,
        and waits for completion.

        :param ConfigManager config: workflow configuration

        :returns: None
        :rtype: None
        """
        from parsl_tasks.gen_structures import gen_structures
        try:
            n_chunks = self.config[CK.GEN_STRUCTURES_NNODES]
            l_futures = [gen_structures(self.config.get_json_config(), n_chunks, i) for i in range(1, n_chunks + 1)]
            _collect_future_errors(l_futures, "generate_structures")

        except Exception as e:
            amd_logger.critical(f"An exception occurred: {e}")

    def not_finished(self) -> bool:
        return not os.path.exists(os.path.join(self.config[CK.WORK_DIR], "structures/1"))

    def run(self) -> None:
        if self.not_finished():
            self._generate_structures()
        amd_logger.info(f"generate structures done")


class CgcnnStep(Step):

    def _run_cgcnn(self):
        """
        Predicts formation energy with CGCNN for the candidates.

        Submits :func:`parsl_tasks.cgcnn.cgcnn_prediction` for each chunk
        ``i ∈ [1, n_chunks]`` where ``n_chunks = config[CK.GEN_STRUCTURES_NNODES]``.
        Merges ``work_dir/test_results_*.csv`` into ``work_dir/test_results.csv``,
        then deletes the shard files.

        :param ConfigManager config: workflow configuration

        :returns: None
        :rtype: None
        """
        from parsl_tasks.cgcnn import cgcnn_prediction
        try:
            n_chunks = self.config[CK.GEN_STRUCTURES_NNODES]
            l_futures = [cgcnn_prediction(self.config.get_json_config(), n_chunks, i) for i in range(1, n_chunks + 1)]
            _collect_future_errors(l_futures, "cgcnn")

            # merge results
            pattern = os.path.join(self.config[CK.WORK_DIR], "test_results_*.csv")
            files_to_merge = list(glob.iglob(pattern))
            if not files_to_merge:
                amd_logger.critical(f"CGCNN: failed to merge into test_results.csv")

            dataframes = pd.concat(
                (pd.read_csv(file, header=None) for file in files_to_merge),
                ignore_index=True
            )
            dataframes.to_csv(os.path.join(self.config[CK.WORK_DIR], "test_results.csv"), index=False)

            # cleanup
            for file in files_to_merge:
                os.remove(file)

        except Exception as e:
            amd_logger.critical(f"An exception occurred: {e}")

    def not_finished(self) -> bool:
        return not os.path.exists(os.path.join(self.config[CK.WORK_DIR], "test_results.csv"))

    def run(self) -> None:
        if self.not_finished():
            self._run_cgcnn()
        amd_logger.info(f"cgcnn done")


class SelectStructuresStep(Step):

    def __init__(self, config, out_dir, min_total=1000, max_total=4000):
        super().__init__(config)
        self.out_dir = Path(self.config[CK.WORK_DIR]) / out_dir
        self.min_total = min_total
        self.max_total = max_total

    def _select_structures(self):
        """
        Filter, deduplicate, and select candidate structures.

        Runs :func:`parsl_tasks.select_structures.select_structures` to produce
        ``{work_dir}/new/POSCAR_{i}`` and ``{work_dir}/new/id_prop.csv``.

        :param ConfigManager config: workflow configuration

        :returns: None
        :rtype: None
        """
        from parsl_tasks.select_structures import select_structures
        try:
            select_structures(self.config.get_json_config(), self.out_dir, self.min_total, self.max_total).result()
        except Exception as e:
            amd_logger.critical(f"An exception occurred: {e}")

    def not_finished(self) -> bool:
        return not (self.out_dir / "POSCAR_1").exists()

    def run(self) -> None:
        if self.not_finished():
            self._select_structures()
        amd_logger.info(f"select structures done")


class MLIPRelaxationStep(Step):

    @staticmethod
    def _extract_index(poscar: Path) -> str:
        """
        Extract structure index from POSCAR filename.
        Example: POSCAR_123.cif -> "123"
        """
        tail = poscar.name.rsplit("_", 1)[-1]
        return tail.split(".", 1)[0]

    def _mlip_relaxation(self):
        from parsl_tasks.mlip_relaxation import mlip_relaxation
        input_dir = Path(self.config[CK.WORK_DIR]) / CK.SELECT_STRUCT_OUTPUT_0
        log_dir = Path(self.config[CK.WORK_DIR]) / CK.MLIP_LOG_DIR
        ener_ml_file = Path(self.config[CK.WORK_DIR]) / CK.MLIP_ENER_ML_FILE
        output_pattern = "CONTCAR"
        num_chunks = self.config[CK.MLIP_RELAX_NNODES] * self.config[CK.GPUS_PER_NODE]

        if ener_ml_file.is_file() and ener_ml_file.stat().st_size > 0:
            amd_logger.info("MLIP relaxation already completed.")
            return

        original_files = sorted(input_dir.glob("POSCAR_*"))
        if not original_files:
            amd_logger.critical(f"Error: No POSCAR files found in {input_dir}")

        to_process = []
        completed = 0

        # check files that were already processed
        for poscar in original_files:
            idx = MLIPRelaxationStep._extract_index(poscar)
            expected_output = log_dir / f"{output_pattern}_{idx}"

            if expected_output.is_file():
                completed += 1
            else:
                to_process.append(poscar)

        if not to_process:
            amd_logger.debug("MLIP relaxation not completed but nothing to run.")
        else:
            # split the work in equal chunks
            files_per_chunk = math.ceil(len(to_process) / num_chunks)
            chunks = [
                to_process[i:i + files_per_chunk]
                for i in range(0, len(to_process), files_per_chunk)
            ]

            # submit tasks (one per chunk)
            futures = []
            for i, chunk in enumerate(chunks):
                file_list = [str(p) for p in chunk]
                futures.append(mlip_relaxation(self.config, file_list))

            for future in futures:
                try:
                    err = future.exception()
                    if err:
                        raise err
                except Exception as e:
                    amd_logger.critical(f"An exception occurred: {e}")

        # write final result
        with open(ener_ml_file, "w") as out:
            for p in sorted(Path(log_dir).glob("energy_*")):
                out.write(p.read_text())

    def not_finished(self) -> bool:
        ener_ml_file = Path(self.config[CK.WORK_DIR]) / CK.MLIP_ENER_ML_FILE
        return not (ener_ml_file.is_file() and ener_ml_file.stat().st_size > 0)

    def run(self) -> None:
        if self.not_finished():
            self._mlip_relaxation()
        amd_logger.info("MLIP relaxation done")


class VaspCalculationsStep(Step):
    def __init__(self, config, run_mlip_post_processing: bool = False):
        super().__init__(config)
        self.run_mlip_post_processing = run_mlip_post_processing

    def _mlip_post_processing(self):
        """
        Prepare structures for VASP optimization after MLIP relaxation.

        This method is used in the MLIP-based workflow to post-process results
        obtained with MLIP relaxation, and prepare a set of POSCAR files for subsequent VASP optimization.
        """
        import shutil
        work_dir = self.config[CK.WORK_DIR]
        structure_dir = os.path.join(work_dir, CK.SELECT_STRUCT_OUTPUT)
        energy_log_dir = os.path.join(work_dir, CK.MLIP_LOG_DIR)
        hull_ml_file = os.path.join(work_dir, CK.MLIP_HULL_ML_FILE)

        if os.path.isfile(os.path.join(structure_dir, "POSCAR_1")):
            return

        os.makedirs(structure_dir, exist_ok=True)
        file_id = 1
        with open(hull_ml_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # skip empty lines

                index, energy = (x.strip() for x in line.split(",", 1))

                orig_file = os.path.join(energy_log_dir, f"CONTCAR_{index}")
                new_file = os.path.join(structure_dir, f"POSCAR_{file_id}")

                if os.path.isfile(orig_file):
                    shutil.copy(orig_file, new_file)
                else:
                    amd_logger.warning(
                        f"Warning: Structure file not found for index {index} ({orig_file}). Skipping."
                    )
                file_id += 1

    def _vasp_calculations(self):
        """
        Run two-stage VASP calculations for all selected structures and log outcomes.

        Launches :func:`parsl_tasks.dft_optimization.run_vasp_calc` for each ID in
        ``{config[CK.VASP_ID_STRUCT_LIST]}``. Writes a CSV of per-ID results to
        ``{config[CK.VASP_WORK_DIR]}/{config[CK.OUTPUT_FILE]}``.

        :param ConfigManager config: workflow configuration

        :returns: None
        :rtype: None

        :raises Exception: only if uncaught errors propagate past per-task handling
         """
        from parsl_tasks.dft_optimization import run_vasp_calc
        work_dir = self.config[CK.WORK_DIR]
        output_file_vasp_calc = os.path.join(
            self.config[CK.VASP_WORK_DIR], self.config[CK.OUTPUT_FILE])

        # open the output file to log the structures that failed or succeded to
        # converge
        fp_mode = "a" if os.path.exists(output_file_vasp_calc) else "w"
        fp = open(output_file_vasp_calc, fp_mode)
        fp.write("id,result\n")

        # launch all vasp calculations
        l_futures = [(run_vasp_calc(self.config.get_json_config(), i), i)
                     for i in self.config[CK.VASP_ID_STRUCT_LIST]]

        # wait for all the tasks (in the batch) to complete
        for future, id in l_futures:
            try:
                err = future.exception()
                if err:
                    raise err
                _write_status(fp, id, "success")
            except tuple(STATUS_BY_EXCEPTION) as e:
                _write_status(fp, id, STATUS_BY_EXCEPTION[type(e)])
            except Exception as e:
                amd_logger.warning(f"An exception occurred: {e}")
                _write_status(fp, id, "unexpected_error")

        fp.close()

    def not_finished(self) -> bool:
        return not os.path.exists(os.path.join(self.config[CK.VASP_WORK_DIR], self.config[CK.OUTPUT_FILE]))

    def run(self) -> None:
        if self.run_mlip_post_processing:
            self._mlip_post_processing()
        self.config.setup_vasp_calculations()
        self._vasp_calculations()
        amd_logger.info(f"vasp calculations done")


class EhullMLParallel(Step):

    def _ehull_ml_parallel(self):
        os.makedirs(self.config[CK.POST_PROCESSING_OUT_DIR], exist_ok=True)
        get_vasp_hull(self.config)

        err = ehull_ml_parallel(self.config).exception()
        if err:
            amd_logger.critical(f"ehull_ml_parallel {err}")

    def not_finished(self) -> bool:
        hull_ml_out = os.path.join(self.config[CK.WORK_DIR], CK.MLIP_HULL_ML_FILE)
        return not (os.path.isfile(hull_ml_out) and os.path.getsize(hull_ml_out) > 0)

    def run(self) -> None:
        if self.not_finished():
            self._ehull_ml_parallel()
        amd_logger.info(f"ehull_ml_parallel done")


class PostProcessingStep(Step):

    def __init__(self, config, get_hull: bool = True):
        super().__init__(config)
        self.get_hull = get_hull

    def _post_processing(self):
        """
        Compute Ehull, color the convex hull, and collect promising candidates.

        Requires a 3- or 4-element system. Runs
        :func:`tools.post_processing.get_vasp_hull`,
        :func:`parsl_tasks.ehull.calculate_ehul` and
        :func:`parsl_tasks.convex_hull.convex_hull_color`.

        :param ConfigManager config: workflow configuration

        :returns: None
        :rtype: None
        """
        if self.config[CK.POST_PROCESSING_OUT_DIR]:

            elements = self.config[CK.ELEMENTS]
            nb_of_elements = len(elements.split('-'))

            if nb_of_elements < 3 or nb_of_elements > 4:
                amd_logger.critical(
                    f"The post-processing is only supported with 3 or 4 elements")

            os.makedirs(self.config[CK.POST_PROCESSING_OUT_DIR], exist_ok=True)

            # for MLIP workflow the hull was compiled in previous steps
            if self.get_hull:
                get_vasp_hull(self.config)

            err = calculate_ehul(self.config).exception()
            if err:
                amd_logger.critical(f"calculate_ehul {err}")

            err = convex_hull_color(self.config).exception()
            if err:
                amd_logger.critical(f"convex_hull_color {err}")

            out_hull = os.path.join(
                self.config[CK.POST_PROCESSING_OUT_DIR], CK.POST_PROCESSING_FINAL_OUT)
            amd_logger.info(f"Convex hull plot saved to '{out_hull}'")

            out_selected = os.path.join(
                self.config[CK.POST_PROCESSING_OUT_DIR], "selected")
            amd_logger.info(f"Selected candidates saved to '{out_selected}'")

    def not_finished(self) -> bool:
        if not self.config[CK.POST_PROCESSING_OUT_DIR]:
            return False

        out_hull = os.path.join(
            self.config[CK.POST_PROCESSING_OUT_DIR], CK.POST_PROCESSING_FINAL_OUT
        )
        return not os.path.exists(out_hull)

    def run(self) -> None:
        if self.not_finished():
            self._post_processing()
        amd_logger.info(f"post-processing done")

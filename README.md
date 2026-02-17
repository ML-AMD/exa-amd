# exa-AMD: Exascale Accelerated Materials Discovery
exa-AMD is a Python framework designed to accelerate the discovery and design of functional materials. The framework uses [Parsl](https://parsl-project.org) to build customizable and automated workflows that connect AI/ML tools, material databases, quantum mechanical calculations, and state-of-the-art computational methods for novel structure prediction. 

exa-AMD is designed to accommodate different workflow styles on high performance computers. It can scale up on supercomputers to use a large number of nodes equipped with accelerators, such as the Nvidia and AMD GPUs. It can also run at a small scale dynamically, coordinating with the queueing system (e.g. Slurm) to automate compute tasks and job submissions. exa-AMD comes with a global registry to support flexible job execution patterns. Users can choose a pre-defined Parsl configuration provided; it is also possible to create a customized Parsl configuration for different computing systems and workflow needs.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Usage](#usage-cli)
- [Register a new Parsl configuration](#registering-a-new-parsl-configuration)
- [Examples](#example)

## Prerequisites
This package requires:
- python >= 3.10
- numpy < 2.0
- scikit-learn >= 1.6.1
- pytorch >= 2.2.2
- torchvision >= 0.17.2
- pymatgen >= 2025.3.10
- parsl >= 2025.3.24
- pytest >= 8.3.5
- sphinx >= 7.1.2
- sphinx_rtd_theme >= 3.0.2
- mp-api >= 0.45.7
- python-ternary >= 1.0.8

Additionally:
- Ensure you have a working [VASP](https://www.vasp.at) installation.
- Ensure you have prepared the initial crystal structures in the Crystallographic Information File (CIF) format and put them in a directory called `initial_structures`. An example of an initial data set can be downloaded [here](https://doi.org/10.5281/zenodo.17180192).
- Create a json file that specifies the running configurations for exa-AMD. See for example [configs/chicoma.json](configs/chicoma.json). The configuration file specifies general settings for running exa-AMD, CGCNN, VASP, and the Parsl configuration.


### External packages 
This package contains a modified version of Crystal Graph Convolutional Neural Networks (CGCNN) placed under the `cms_dir` directory. The original [CGCNN](https://github.com/txie-93/cgcnn) source code was developed by [Tian Xie](https://txie.me/) et al., distributed under the MIT License.

### Workflow-dependent requirements

Some workflows require additional resources beyond the general prerequisites.

- **MLIP workflow**: Requires the pretrained model checkpoint `ml_models/mlip/uma-s-1p1.pt`. This file is managed using [Git Large File Storage (Git LFS)](https://git-lfs.com/). To ensure the model weights are properly retrieved, Git LFS must be installed and initialized before cloning the repository:

  ```bash
  git lfs install
  ```
## Install
### CLI (preferred)
```bash
# (Recommended) create & activate a fresh virtual environment
# python -m venv myenv
# source myenv/bin/activate
# python -m pip install --upgrade pip
pip install "https://github.com/ML-AMD/exa-amd/releases/download/v0.1.0/exa_amd-0.1.0-py3-none-any.whl"
exa_amd --help
```
### Conda-only (run from source)
If you use [Conda](https://docs.conda.io/en/latest/miniconda.html) to manage Python packages, you can create the environment using the file included in this repository: [amd_env.yml](./amd_env.yml).
```bash
conda env create -f amd_env.yml
conda activate amd_env
# from the repo root:
python exa_amd.py --config your_config.json
```

## Structure Input Formats

exa-AMD utilizes the [`pymatgen`](https://pymatgen.org) library for handling atomic structures. The `pymatgen.Structure` class supports a wide range of structure file formats, including *CIF*, and *POSCAR*.

In the current implementation, *CIF* files are used as the default input format for the initial structure pool provided with the package. However, users are free to supply structures in any format readable by `pymatgen`.

## Registering a new Parsl configuration
We currently support the automated workflows on NERSC's Perlmutter and LANL's Chicoma computers. If you would like to run on a different computing system, you must add your own Parsl configuration following these steps:

1. Create a new file in your configs directory (e.g., `parsl_configs/`), similar to the one in [parsl_configs/chicoma.py](parsl_configs/chicoma.py)
2. Add a configuration class with a unique name `<my_parsl_config_name>` 
3. Modify Parsl's execution settings. More details can be found in [Parsl's official documentation](https://parsl.readthedocs.io/en/latest/)
4. Register your configuration class by calling `register_parsl_config()` inside that file
5. Modify your json configuration file accordingly by setting `parsl_config` to `<my_parsl_config_name>` and `parsl_configs_dir` to the absolute path of your configs directory (e.g., `<abs_path_to>/parsl_configs`)


## Usage (CLI)
- Modify the `initial_structures_dir` field in the json configuration file to indicate the absolute path to the `initial_structures` directory

- Modify the `parsl_configs_dir` field in the json configuration file to indicate the absolute path to the Parsl configurations (e.g., `<abs_path>/parsl_configs`)

- Run exa-AMD with the json file created in the prerequisite step:
    ```bash
    exa_amd --config <your_config_file>
    ```
    For running on Perlmutter for example,
    ```bash
    exa_amd --config configs/perlmutter.json
    ```
- (Optional) The json config file can be overridden via command line arguments, for example:
    ```bash
    exa_amd --config configs/perlmutter.json --num_workers 256
    ```
    For a full list of command line arguments and their descriptions, run:
    ```bash
    exa_amd --help
    ```

## Documentation
Documentation for exa-AMD is at [https://ml-amd.github.io/exa-amd](https://ml-amd.github.io/exa-amd/).

## Example
Follow the step-by-step tutorial: [exa-AMD Tutorial](https://ml-amd.github.io/exa-amd/tutorial.html).

## Highlight
Prediction of new CeFeIn compounds using this framework by the development team.

<img width="677" alt="thrust1" src="https://github.com/user-attachments/assets/b067d23f-fd43-4409-b44b-01d1457bb440" />

## How to cite

If you use exa-AMD, please cite the following papers:

- [JOSS](https://joss.theoj.org/papers/10.21105/joss.08879)
- [arXiv:2510.01170](https://arxiv.org/abs/2510.01170)  

If you configure exa-AMD to make use of CGCNN for structure screening, please also cite the original CGCNN paper:  
[https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301)

## Contribute
- **Propose changes** — [open a pull request](https://github.com/ML-AMD/exa-amd/pulls)
- **Report issues** — [open an issue](https://github.com/ML-AMD/exa-amd/issues)
- **Ask questions / get help** — [start a discussion](https://github.com/ML-AMD/exa-amd/discussions)

## Copyright
Copyright 2025. Iowa State University. All rights reserved. This software was produced under U.S. Government contract DE-AC02-07CH11358 for the Ames National Laboratory, which is operated by Iowa State University for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software. NEITHER THE GOVERNMENT NOR IOWA STATE UNIVERSITY MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from the Ames National Laboratory.

© 2025. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.(Copyright request O4873).

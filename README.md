# exa-AMD: Exascale Accelerated Materials Discovery
exa-AMD is a Python framework designed to accelerate the discovery and design of functional materials. The framework uses [Parsl](https://parsl-project.org) to build customizable and automated workflows that connect AI/ML tools, material databases, quantum mechanical calculations, and state-of-the-art computational methods for novel structure prediction. 

exa-AMD is designed to accommodate different workflow styles on high performance computers. It can scale up on supercomputers to use a large number of nodes equipped with accelerators, such as the Nvidia and AMD GPUs. It can also run at a small scale dynamically, coordinating with the queueing system (e.g. Slurm) to automate compute tasks and job submissions. exa-AMD comes with a global registry to support flexible job execution patterns. Users can choose a pre-defined Parsl configuration provided; it is also possible to create a customized Parsl configuration for different computing systems and workflow needs.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Register a new Parsl configuration](#register-parsl-config)
- [Examples](#examples)

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

If you use [Conda](https://docs.conda.io/en/latest/miniconda.html) to manage Python packages, you may create a conda environment to install the required packages using the `amd_env` environment yaml file we provide:
```bash
conda env create -f amd_env.yml
conda activate amd_env
```

Additionally:
- Ensure you have a working [VASP](https://www.vasp.at) installation.
- Ensure you have prepared the initial crystal structures in the Crystallographic Information File (CIF) format and put them in a directory called `initial_structures`. An example of an initial data set can be downloaded [here](https://iastate.box.com/s/3swro78kbcd69fwamhk7df4n5rx4edae).
- Create a json file that specifies the running configurations for exa-AMD. See for example [configs/chicoma.json](configs/chicoma.json). The configuration file specifies general settings for running exa-AMD, CGCNN, VASP, and the Parsl configuration.


### External packages 
This package contains a modified version of Crystal Graph Convolutional Neural Networks (CGCNN) placed under the `cms_dir` directory. The original [CGCNN](https://github.com/txie-93/cgcnn) source code was developed by [Tian Xie](https://txie.me/) et al., distributed under the MIT License.


## Registering a new Parsl configuration
We currently support the automated workflows on NERSC's Perlmutter and LANL's Chicoma computers. If you would like to run on a different computing system, you must add your own Parsl configuration following these steps:

1. Create a new file in the `parsl_configs` directory, similar to the one in [parsl_configs/chicoma.py](parsl_configs/chicoma.py)
2. Add a configuration class with a unique name `<my_parsl_config_name>` 
3. Modify Parsl's execution settings. More details can be found in [Parsl's official documentation](https://parsl.readthedocs.io/en/stable/userguide/configuration/execution.html)
4. Register your configuration class by calling `register_parsl_config()`
5. Modify your json configuration file accordingly by setting `parsl_config` to `<my_parsl_config_name>`


## Usage
- Modify the `initial_structures_dir` field in the json configuration file to indicate the absolute path to the `initial_structures` directory

- Run exa-AMD with the json file created in the prerequisite step:
    ```bash
    python exa_amd.py --config <your_config_file>
    ```
    For running on Perlmutter for example,
    ```bash
    python exa_amd.py --config configs/perlmutter.json
    ```
- (Optional) The json config file can be overridden via command line arguments, for example:
    ```bash
    python exa_amd.py --config configs/perlmutter.json --num_workers 256
    ```
    For a full list of command line arguments and their descriptions, run:
    ```bash
    python exa_amd.py --help
    ```

## Documentation
Documentation for exa-AMD is at [https://ml-amd.github.io/exa-amd](https://ml-amd.github.io/exa-amd/).

## Example
Follow the step-by-step tutorial: [exa-AMD Tutorial](https://ml-amd.github.io/exa-amd/tutorial.html).

## Highlight
Prediction of new CeFeIn compounds using this framework by the development team.

<img width="677" alt="thrust1" src="https://github.com/user-attachments/assets/b067d23f-fd43-4409-b44b-01d1457bb440" />

## Contribute
- **Propose changes** — [open a pull request](https://github.com/ML-AMD/exa-amd/pulls)
- **Report issues** — [open an issue](https://github.com/ML-AMD/exa-amd/issues)
- **Ask questions / get help** — [start a discussion](https://github.com/ML-AMD/exa-amd/discussions)

## Copyright
Copyright 2025. Iowa State University. All rights reserved. This software was produced under U.S. Government contract DE-AC02-07CH11358 for the Ames National Laboratory, which is operated by Iowa State University for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software. NEITHER THE GOVERNMENT NOR IOWA STATE UNIVERSITY MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from the Ames National Laboratory.

© 2025. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.(Copyright request O4873).

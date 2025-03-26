# exa-AMD
exa-AMD is a Python framework designed to accelerate the discovery and design of functional materials. The framework uses [Parsl](https://parsl-project.org) to build customizable and automated workflows that connect AI/ML tools, material databases, quantum mechanical calculations, and state-of-the-art computational methods for novel structure prediction. exa-AMD was designed to scale up on high-performance computing systems including supercomputers equipped with accelerators, such as the Nvidia and AMD GPUs.

It comes with a flexible configuration system based on a global registry. You can choose which Parsl configuration to load at runtime by setting the `parsl_config` key in the global configuration.It is also possible to create new configs simply by creating a new file in the `parsl_configs` directory and registering it.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Register a new Parsl config](#register-parsl-config)
- [Examples](#examples)

## Prerequisites
- Ensure you have a working VASP installation
- Ensure you have prepared the initial crystal structures in the Crystallographic Information File (CIF) format and put in a directory called `mpstrs` 
```bash
cp -R mpstrs ./ctest 
```
- Ensure you have [Conda](https://docs.conda.io/en/latest/miniconda.html) installed.
- Install the required packages and activate the `amd_env` environment
```bash
conda env create -f amd_env.yml
conda activate amd_env

```

## Usage
- Copy the `mpstrs` directory into the `cms_dir`
- Set up a json configuration file (similar to [configs/chicoma.json](configs/chicoma.json))
- Run `python amd.py --config <your_config_file>`

The json config file can be overriden via command line arguments.

```bash
python amd.py --help
```

## Register a new Parsl config
Add your own Parsl config (if we do not support your system)

1. Create a new config similar to [parsl_configs/chicoma.py](parsl_configs/chicoma.py)
2. Register you config by calling `register_parsl_config()` and choosing an unique name `<my_parsl_config_name>`
3. Modify your json config accordingly (i.e. set `parsl_config` to `<my_parsl_config_name>`)

## Examples
Prediction of new CeFeIn compounds using this framework.

<img width="677" alt="thrust1" src="https://github.com/user-attachments/assets/b067d23f-fd43-4409-b44b-01d1457bb440" />




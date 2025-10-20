---
title: 'exa-AMD: A Scalable Workflow for Accelerating AI-Assisted Materials Discovery and Design'
tags:
  - Machine learning
  - Material databases
  - Heterogeneity
  - HPC workflows
authors:
  - name: Maxim Moraru
    affiliation: 1
  - name: Weiyi Xia
    affiliation: 2
  - name: Zhuo Ye
    affiliation: 2
  - name: Feng Zhang 
    affiliation: 2
  - name: Yongxin Yao
    affiliation: 2
  - name: Ying Wai Li
    affiliation: 1
  - name: Cai-Zhuang Wang 
    affiliation: 2

affiliations:
  - name: Los Alamos National Laboratory, Los Alamos, NM 87545, USA
    index: 1
  - name: Ames Laboratory, US DOE and Department of Physics and Astronomy, Iowa State University, Ames, Iowa 50011, United States
    index: 2
date: June 25, 2025
bibliography: "exa_amd_joss.bib"
---

# Summary
exa-AMD is a Python-based application designed to accelerate the discovery and design of functional materials by integrating AI/ML tools, materials databases, and quantum mechanical calculations into scalable, high-performance workflows. The execution model of exa-AMD relies on Parsl [@babuji2019parsl], a task-parallel programming library that enables a flexible execution of tasks on any computing resource from laptops to supercomputers. exa-AMD provides the following key-features:

- **Modularity:** The workflow is composed of interchangeable task modules. New data sources, machine-learning models, or post-processing stages can be added or replaced while leaving the rest of the workflow unchanged.
- **Scalability:** exa-AMD scales efficiently from a single workstation to many supercomputer nodes. It demonstrated near-linear speedup on up to 1024 GPUs and 4,096 CPUs through Parsl's dynamic task distribution, as detailed in our benchmarking study [@xiaa2025exaamd].
- **Elasticity:** computing resources can be added or released at run time, allowing the workflow to exploit shared supercomputers efficiently and assign dynamically specialized accelerators (e.g., GPUs) to different tasks.
- **Resumability\*:** The workflow is divided into fine-grained tasks, allowing exa-AMD to track completed steps so that subsequent runs can resume from where it left off. [^resumability]
- **Configurability:** exa-AMD exposes high-level configuration parameters to allow the users to balance performance and accuracy for their scientific objectives. In particular, a JSON configuration file expose parameters for: ML model selection, energy thresholds, DFT convergence criteria, structural similarity cutoffs, and computing resource allocation. 

exa-AMD is specifically designed for crystalline inorganic materials discovery, including metals, intermetallics, ceramics, and semiconductors operating under periodic boundary conditions. Unlike general-purpose workflow management systems such as AiiDA [@PIZZI2016218] or atomate2 [@ganose2025atomate2], exa-AMD provides a domain-specific, end-to-end workflow tailored for AI-assisted materials discovery. Its modular design allows users to extend or replace components (e.g., machine learning models, data sources, or post-processing utilities), but its core goal is to deliver a pre-configured and scalable research pipeline optimized for materials discovery of multinary systems rather than serving as an materials modeling automation framework.

[^resumability]: *Scope of resumability.* exa-AMD does not implement classical checkpoint/restart. Instead, resumability arises from the workflow’s modular design: at runtime, exa-AMD detects the last completed phase and starts with the next one. For DFT stages, partially processed structure sets also resume automatically. For example, if a user processes the first 100 structures in one run, the next run begins at structure 101.

![Example output from an exa-AMD workflow for the Ce-Fe-In chemical system. (Left) The predicted ternary convex hull, where each point represents a calculated crystal structure. The color indicates the energy above the hull (E_hull), with points on or very near the hull (in red, E_hull ≈ 0) predicted to be thermodynamically stable or metastable. (Right) The crystal structure of Ce₃FeIn, a new ternary compound identified by the workflow as a promising stable phase, located on the convex hull. This figure demonstrates the capability of exa-AMD to perform high-throughput screening and identify novel, potentially synthesizable materials.](CeFeIn_prediction.png){ width=100%}

# Statement of Need
High-performance functional materials are critical for advanced technology innovation and sustainable development. However, the pace of discovery and design of novel functional materials is far behind technological demands. For example, only ~200,000 inorganic crystalline compounds are experimentally known despite theoretical estimates suggesting millions of thermodynamically stable compositions remain undiscovered [@Merchant2023]. The development of next-generation batteries, permanent magnets, and catalysts requires exploring complex multinary (3+ element) systems where combinatorial explosion makes exhaustive experimental synthesis impractical.

Traditional materials discovery workflows require researchers to manually: (1) identify candidate structures from databases or literature, (2) prepare input files for DFT calculations, (3) submit jobs to computing clusters with custom job scripts, (4) monitor calculations and handle failures, (5) extract results from output files, (6) perform post-processing analysis in separate scripts, and (7) manually construct phase diagrams. This fragmented process is time-consuming, error-prone, and difficult to scale beyond a few dozen candidate structures. exa-AMD automates this entire pipeline, enabling exploration of thousands of candidates with minimal user intervention.

exa-AMD addresses this need by providing a modular and configurable platform that connects multiple computational techniques specific to materials discovery in a unified workflow. It supports heterogeneous execution across multiple nodes types and enables high-throughput processing of structure candidates. By using Parsl, exa-AMD is able to decouple the workflow logic from execution configuration, thereby empowering researchers to scale their workflows without having to reimplement them for each system.

# Workflow Overview
exa-AMD implements a five-stage workflow as illustrated in Figure 2. Each stage may initiate multiple asynchronous tasks that Parsl distributes across available computing resources (nodes, cores, or accelerators). Within each task, computations may leverage shared-memory parallelism (e.g., VASP's MPI+OpenMP on CPUs, shown in blue) or GPU acceleration (e.g., CGCNN inference, shown in green). This two-level parallelism enables efficient utilization of heterogeneous HPC systems.

The workflow starts with the generation of hypothetical crystal structures based on the initial templates provided by the user. In this step, target elements are substituted into existing crystal structures, creating chemically plausible candidates for further analysis. While current implementation uses template-based substitution, the structure generation stage is modular and can be replaced with: Random structure generation (e.g., via USPEX-style methods), Genetic algorithms (e.g., AGA, CALYPSO, etc), Prototype enumeration (e.g., AFLOW prototype library), User-provided candidate sets.In the next stage, the formation energies of the generated candidates are predicted using a Crystal Graph Convolutional Neural Network (CGCNN) model [@Xie2018]. Structures with low predicted formation energies are selected as promising candidates for further study. This step enables high-throughput screening and prioritization, reducing the computational cost of subsequent first-principles calculations. Similarly to previous step, the CGCNN module is designed as a pluggable component. Alternative models can be integrated by implementing the same interface, such as ALIGNN, M3GNet, MEGNet, CHGNet, etc. Following CGCNN screening, a filtering stage removes duplicate or near-duplicate structures, based on a structural similarity threshold. Then, the filtered set of structures is subjected to first-principles calculations using Density Functional Theory (DFT), as implemented in the VASP package [@Kresse1996a;@Kresse1996b].

After the completion of VASP calculations, exa-AMD performs automated post-processing to extract and analyze key physical properties from the calculation outputs. This final stage computes the formation energies of each structure relative to reference elemental phases, which are then used for constructing the convex hull—the set of thermodynamically stable phases at zero temperature and pressure, where any compound above this hull is metastable or unstable [@ong2008li]. Structures with energy above the convex hull below a user-configurable threshold (typically 0.1 eV/atom) are identified as promising candidates and are automatically copied to a dedicated folder for further analysis. (Note that this criterion, based on thermodynamic stability relative to the convex hull, differs from the one used in the structure selection stage, where candidates are filtered based on their absolute formation energies, e.g., below a threshold of −0.2 eV/atom.) At the end of this stage, exa-AMD generates an updated phase diagram by plotting the convex hull. An example for the Ce-Fe-In system is shown in Figure 1, where exa-AMD identified several new potential compounds on the convex hull, including Ce₃FeIn.


![The core five-stage workflow implemented in exa-AMD. This represents a pre-configured pipeline for AI-assisted materials discovery. While this is the default workflow, the modular design of exa-AMD allows users to customize, replace, or extend individual stages to suit different research objectives. Inputs include target chemical elements and initial structure templates. Intermediate data includes ML-predicted formation energies and structural similarity metrics. Outputs comprise DFT-optimized structures, formation energies, energy above convex hull values, and updated phase diagrams.](workflow.png){ width=60%}


# Initial Crystal Structures
exa-AMD requires an initial set of crystal structures used as starting points in the workflow. For investigations involving any multinary system, the input dataset can be populated with any relevant set of initial structures, such as quaternary prototypes, user-defined entries, or structures taken from one or multiple database sources including but not limited to Materials Project [@Jain2013], GNoME [@Merchant2023], AFLOW [@Curtarolo2012], OQMD [@Saal2013;@Kirklin2015], etc. This flexibility makes the workflow adaptable to a wide range of compositional and structural spaces.

# Acknowledgements
This work was supported by the U.S. Department of Energy (DOE), Office of Science, Basic Energy Sciences, Materials Science and Engineering Division through the Computational Material Science Center program. Ames National Laboratory is operated for the U.S. DOE by Iowa State University under contract # DE-AC02-07CH11358. Los Alamos National Laboratory is operated by Triad National Security, LLC, for the National Nuclear Security Administration of U.S. Department of Energy under Contract No. 89233218CNA000001.

This research used resources provided by the National Energy Research Scientific Computing Center, supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231, and resources provided by the Los Alamos National Laboratory Institutional Computing Program.

Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the U.S. Department of Energy’s National Nuclear Security Administration.

LA-UR-25-26122

# References

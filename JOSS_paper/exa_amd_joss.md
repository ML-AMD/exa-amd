---
title: 'exaAMD: Exascale Framework for Accelerating AI-Assisted Materials Discovery and Design'
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
  - name: Ying Wai Li
    affiliation: 1
  - name: Cai-Zhuang Wang 
    affiliation: 2
  - name: Other co-authors to be added ...
    affiliation: 2
  
affiliations:
  - name: Los Alamos National Laboratory, Los Alamos, NM 87545, USA
    index: 1
  - name: Ames Laboratory, US DOE and Department of Physics and Astronomy, Iowa State University, Ames, Iowa 50011, United States
    index: 2
date: May 15, 2025
bibliography: "exa_amd_joss.bib"
---


# Summary

exa-AMD is a Python-based workflow framework designed to accelerate the discovery and design of functional materials by integrating AI/ML tools, materials databases, and quantum mechanical calculations into scalable, high-performance workflows. The exacution model of exa-AMD relies on Parsl [@babuji2019parsl], a task-parallel programming library that enables the execution of distributed and heterogeneous workflows. exa-AMD is optimized for GPU-based supercomputing environments and has been successfully deployed on large-scale HPC systems.

![Prediction of new CeFeIn compounds.](CeFeIn_prediction.png){ width=100%}

# Statement of Need

Materials discovery remains a time-consuming and computationally expensive process. While the community has access to high-quality simulation tools, machine learning models, and materials databases, integrating these components into a cohesive and scalable workflow remains a challenge, especially on large-scale systems. 

exa-AMD addresses this need by providing a modular and configurable framework that connects multiple computational techniques specific to materials discovery in a unified workflow. The framework supports heterogeneous execution across multiple node types, allows high-throughput processing of structure candidates, and is optimized for deployment on GPU-powered HPC systems. By using Parsl, exa-AMD is able to decouple the workflow logic from execution configuration, and therefore it empowers researchers to scale their workflows without having to reimplement them for each system.

# Workflow Overview

The framework proposes a four-stage pipeline that includes structure generation, ML-based structure screening, structure selection, and density functional theory (DFT) calculations. These stages are implemented as separate Parsl tasks and executed in parallel using multiple executors across CPU and GPU resources.

The proposed workflow begins with the generation of hypothetical crystal structures. In this step, target elements are substituted into existing crystal structures, creating chemically plausible candidates for further analysis. To account for all possible atomic arrangements, the code randomly shuffles the order of substituted elements. Lattice scaling is applied, typically ranging from 0.94 to 1.06, to ensure that the generated structures cover a realistic range of bond lengths. This addresses the fact that the ideal bond lengths for the new elements may differ significantly from those in the original structure. The combination of element shuffling and lattice scaling results in a large set of hypothetical structures. For example, in a ternary system, six possible element orderings and five scaling factors yield 30 variants per initial structure. For a quaternary system, there would be 24 possible element orderings.

Once the hypothetical structures are generated, the next stage involves evaluating them using a Crystal Graph Convolutional Neural Network (CGCNN) model [17], which efficiently predicts their formation energies. Structures with low predicted formation energies are selected as promising candidates for further study. This step enables high-throughput screening and prioritization, reducing the computational cost of subsequent calculations.

Following CGCNN screening, a filtering stage removes duplicate or near-duplicate structures , based on a structural similarity threshold. This deduplication step ensures that only non-equivalent structures are retained, typically narrowing the set to a manageable number (e.g., 1,000–5,000 structures) for detailed study.

Finally, the filtered set of structures is subjected to first-principles calculations using Density Functional Theory (DFT), with the VASP package [18-19] in our current GPU-enabled version of code (can be extended to other ab initio code like QUANTUM ESPRESSO [20-21], etc). Each structure undergoes full relaxation to find its lowest-energy configuration, followed by a self-consistent total energy calculation. The resulting relaxed structures and energies provide an accurate basis for subsequent thermodynamic analysis.

# Input data
exa-AMD requires an initial set of crystal structures used as starting points in the workflow. The dataset used in our previous work [1-11] contained ternary structures sourced from the Materials Project database [12]. However, the approach is general: for investigations involving any multinary systems, the input dataset can be populated with any relevant set of initial structures, including quaternary prototypes or user-defined entries, and from one or multiple database sources (including but not limited to Materials Project [12], GNoME [13], AFLOW [14], OQMD [15-16], etc). This flexibility allows the workflow to be adapted to a wide range of compositional and structural spaces.

# Acknowledgements
Parsl [@babuji2019parsl]


# References
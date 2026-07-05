# SpectraLift: Physics-Guided Spectral-Inversion Network for Hyperspectral Image Super-Resolution

**Authors:** Ritik Shah ([ritik.shah@aritheon.com](mailto:ritik.shah@aritheon.com)), Marco Duarte ([mduarte@ecs.umass.edu](mailto:mduarte@ecs.umass.edu))  
Published at IEEE Workshop on Hyperspectral Imaging and Signal Processing: Evolution in Remote Sensing (WHISPERS) 2025.

📄 [Extended version on arXiv](https://arxiv.org/pdf/2507.13339)

---

## 🔍 Overview

Welcome to the official implementation of **SpectraLift**. This repository accompanies our research paper and includes code, notebooks, datasets, precomputed results, and environment files for experimentation and analysis.

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Environment Setup](#-environment-setup)
- [Datasets](#-datasets)
- [Reproducibility](#-reproducibility)
- [Citation](#-citation)
- [License](#-license)

---

## 📁 Repository Structure

### 🔸 CSV Files with Metrics

- Organized into subfolders by synthetic HSI dataset:
  - *Botswana*
  - *Kennedy Space Center*
  - *Pavia Center*
  - *Pavia University*
  - *Washington DC Mall*
- Each subfolder contains:
  - CSVs with image quality metrics across various experimental configurations (e.g., different PSFs, downsampling ratios, and MSI band counts).
  - An Excel summary: `All SOTA methods comparison.xlsx` aggregates all individual method metrics and comparison tables
    - Includes the sheet `'Individual metrics analysis'`, which:
      - Shows how often each method achieves the best metric per dataset.
      - Summarizes performance trends and strengths of different methods.

---

### 🔸 Comparison Implementations

Benchmarks against state-of-the-art supervised and unsupervised baselines.

- **Supervised**  
  - Pre-executed Jupyter notebooks for: `FeINFN`, `FusFormer`, `GuidedNet`, `MIMO-SST`
  - Applied to both synthetic datasets and the University of Houston dataset.

- **Unsupervised**  
  - Pre-executed Jupyter notebooks for: `MIAE`, `C2FF`, `SSSR`, `SDP`
  - Applied to both synthetic datasets and the University of Houston dataset.

---

### 🔸 SpectraLift Implementation
  > Note: The Spectral Inversion Network (SIN) mentioned in the paper is called SpectralSR_MLP in the code on this repository. 

- **SpectraLift Implementation** (`Spectralift_Implementation_Jupyter_Notebooks`):  
  Pre-executed notebooks demonstrating SpectraLift on both synthetic and University of Houston datasets.

- **Ablation Study** (`Spectralift_Ablation_Study_Jupyter_Notebooks`):  
  Pre-executed notebooks evaluating contributions of different model components and architectural choices.

- **Python Scripts** (`SpectraLift_python`):  
  Full Python-based implementation for use in IDEs or via command line.  
  Detailed instructions available in [Python Implementation Details](py_implementation.md).

- **Visualization Notebooks** (`Super_Resolved_Images_Comparison_Jupyter_Notebooks`):  
  Pre-executed notebooks for visual comparison of high-resolution HSI outputs of all methods on each dataset. Includes:
  - Spectral plots
  - Image visualizations
  > Note: Botswana output is split across two notebooks due to data size.

---

## ⚙️ Environment Setup

Use either `pip` or `conda` to recreate the project environment:
> Note: This environment contains all the packages necessary to execute all the jupyter notebooks and python files in this repository, including the comparison methods.

### Using pip:
```bash
pip install -r requirements.txt
```
### Using Conda:
```bash
conda env create -f spectralift-env.yaml
conda activate spectralift-env
```
In rare cases, the Conda YAML file may fail to recreate the environment correctly. If so, use requirements.txt. In the case where both the yaml as well as requirements.txt fail to create the environment using the commands given above, please follow the instructions in the [Manual Environment Setup Guide](ENVIRONMENT_SETUP.md) to be able to create a conda environment with every dependency you need to execute any file within this repository.

## 🗂️ Datasets

SpectraLift uses the following datasets for all synthetic experiments which are easily available publically:

- Washington DC Mall  
- Kennedy Space Center  
- Pavia University  
- Pavia Center  
- Botswana  

### 📥 University of Houston Dataset

The full University of Houston dataset can be downloaded from the official source:

🔗 [2018 IEEE GRSS Data Fusion Challenge](https://machinelearning.ee.uh.edu/2018-ieee-grss-data-fusion-challenge-fusion-of-multispectral-lidar-and-hyperspectral-data/)

---

## ✅ Reproducibility

To ensure full transparency and ease of experimentation:

- All notebooks are **pre-executed**, so results can be viewed immediately without re-running code.
- Environment files are provided to replicate the exact Python environment used in our experiments.
- Python scripts are included along with detailed instructions to ensure that researchers can use our method with ease.

---

## 🛡️ Contribution Guidelines
This is a public, read-only repository. If you'd like to suggest changes:
- Please **fork** the repo
- Make your edits
- Submit a **pull request** for review

Direct pushes are disabled. Thanks for respecting this workflow!

## 📖 Citation

If you use this repository or build upon our work, please cite the following paper:

```bibtex
@misc{spectralift,
  title={SpectraLift: Physics-Guided Spectral-Inversion Network for Self-Supervised Hyperspectral Image Super-Resolution},
  author={Ritik Shah and Marco F. Duarte},
  year={2025},
  eprint={2507.13339},
  archivePrefix={arXiv},
  primaryClass={eess.IV},
  url={https://arxiv.org/abs/2507.13339}
}
```

## 📜 License

This repository is licensed under the **MIT License**.

You are free to use, modify, and distribute this code with proper attribution.

For full details, see the [LICENSE](./LICENSE) file in the repository.

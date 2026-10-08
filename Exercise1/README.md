# Protein Sequence Design and Structure Evaluation Tutorial

This project uses PLP-dependent alanine racemase (PDB: `1SFT`, chain A) as an example to demonstrate protein sequence design with ProteinMPNN and ESM3, followed by structure prediction and comparison with ESMFold.

## 1. Environment Setup

We recommend creating an isolated Conda environment with Python 3.10.

```bash
conda create -n protein-design python=3.10 -y
conda activate protein-design
```

After entering the project directory, install the dependencies listed in `requirements_exercise1.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements_exercise1.txt
```

Then install Matplotlib and the specified versions of PyTorch, TorchVision, and TorchAudio:

```bash
pip install matplotlib torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 accelerate==0.26.0
```


ESM3 and ESMFold are large models that require substantial storage and GPU memory. To avoid storing duplicate copies under every account, their model files have been deployed under another user account on the server. The notebooks load these shared model files directly through the absolute paths defined in the code, so users do not need to download them again.

Before running the notebooks, make sure that your account has read permission for the shared model directories and check the following paths in the code:

- the ProteinMPNN source code and model-weight directories;
- the shared ESM3 model directory;
- the shared ESMFold model directory;
- the project, input PDB, output FASTA, and result directories.

The example target structure is located at:

```text
data/target.pdb
```

## 2. Notebook Overview

### `1.ProteinMPNN_design.ipynb`

This notebook uses ProteinMPNN to generate new protein sequences for the alanine-racemase backbone. It loads and visualizes the target structure, compares designs generated at different sampling temperatures, and demonstrates how to preserve PLP-binding and catalytic-site residues. The designed sequences and their scores are saved for downstream evaluation.

### `2.ESM3_sequence_design.ipynb`

This notebook uses the shared local ESM3-small model for several sequence-design tasks: unconditional generation, local sequence redesign, backbone-conditioned inverse folding, and function-conditioned generation. The generated candidates are collected and saved in CSV and FASTA formats.

### `3.Structure_prediction_comparison.ipynb`

This notebook selects a high-ranking ProteinMPNN design and predicts its three-dimensional structure with the shared local ESMFold model. It then aligns the prediction with the experimental target backbone, calculates pLDDT, pTM, C-alpha RMSD, and per-residue deviations, and visualizes the predicted structure together with the target structure and PLP cofactor.

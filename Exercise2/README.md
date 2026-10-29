# Protein Filtering Tutorial

This graduate practical for the **Protein Design** course uses three PLP-dependent alanine racemases to demonstrate protein filtering. It predicts structures with ESMFold, examines pLDDT, scores solubility with GATSol, predicts kcat for L-alanine with UniKP, and checks the expected protein function through the InterPro website.

All three models use the same sequences from `plp_sequences.csv`. UniProt IDs connect the predicted structures, scores, and functional annotations.

## 1. Environment Setup

We recommend creating an isolated Conda environment with Python 3.10.

```bash
conda create -n protein-design_ex2 python=3.10 -y
conda activate protein-design_ex2
```

After entering the project directory, install the dependencies listed in `requirements_ex2.txt`:

```bash
pip install -r requirements_ex2.txt
```


## 2. Notebook Overview

### [`Protein_Filtering.ipynb`](Protein_Filtering.ipynb)

The notebook processes all three input sequences in the following order:

1. **Load the candidates.** Read the sequences and retain their UniProt IDs.
2. **Predict structures with ESMFold.** Save a PDB for each protein, summarize pLDDT, and inspect confidence profiles and interactive structure views.
3. **Predict solubility with GATSol.** Pass each sequence and its matching ESMFold PDB to `predict_sol.py`, then rank the proteins by their solubility scores.
4. **Predict kcat with UniKP.** Pair each sequence with L-alanine SMILES, predict turnover numbers in s⁻¹, and save the ranking and combined model results. PLP is the cofactor; L-alanine is the substrate used for this prediction.
5. **Inspect function with InterPro.** Export the same sequences as FASTA and submit them through the website. Examine protein families, domains, and functional sites for support of alanine racemase function.

### Input sequences

[`plp_sequences.csv`](plp_sequences.csv) contains full UniProt canonical sequences, protein names, organisms, source URLs, lengths, and sequence MD5 checksums.

| UniProt ID | Organism | Residues |
|---|---|---:|
| P10724 | Geobacillus stearothermophilus | 388 |
| P0A6B4 | Escherichia coli K12 | 359 |
| P10725 | Bacillus subtilis 168 | 389 |

### Supporting code

- [`predict_sol.py`](predict_sol.py) provides the GATSol inference used by the notebook.
- [`UniKP/UniKP_kcat.py`](UniKP/UniKP_kcat.py) provides `predict_kcat()` and loads the local UniKP weights and shared ProtT5 model.

## 3. Running the Tutorial

Activate your prepared course environment. For example, if it is named `protein-design_ex2`, run the following commands from the project root:

```bash
conda activate protein-design_ex2
cd Exercise2
jupyter lab
```

Open `Protein_Filtering.ipynb`, select the appropriate kernel, check the model paths in the first configuration cell, and run the cells from top to bottom. Reduce `CHUNK_SIZE` from 64 to 32 or 16 if ESMFold needs less GPU memory.

The InterPro step is completed manually in the website. Structure HTML files require an internet connection to load 3Dmol JavaScript. Restart the kernel after changing inputs or configuration. Repeated runs overwrite files with the same names; change `OUTPUT_DIR` to preserve earlier results.

After the notebook has generated the first protein's PDB, you can also test GATSol directly from `tt2` using its configured environment:

```bash
python predict_sol.py
```

This command uses P10724 from `plp_sequences.csv`, its predicted PDB in the default output directory, and the model paths defined in `predict_sol.py`.

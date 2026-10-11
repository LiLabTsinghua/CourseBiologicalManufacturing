"""Predict GATSol sol from a protein sequence (or a single-record FASTA file) and a PDB.

Usage examples (in an existing GATSol environment):
    python predict_sol.py  # Test the first CSV sequence and its ESMFold-predicted structure.
    python predict_sol.py protein.fasta protein.pdb
    python predict_sol.py MEVYKKQL... protein.pdb --weights best_model.pt

Dependencies: torch, torch-geometric, fair-esm, biopython, and numpy.
The CLI uses the local GATSol and ESM-1b weights configured below by default; override them with arguments.
The no-argument test requires a predicted PDB in outputs/section2_protein_filtering/structures.
Requirements: 20 standard amino acids only, a sequence length of 1–1022, and a complete matching chain.
The output is the original model's regression value, without sigmoid, normalization, or clipping.
"""

import argparse
import csv
import os
from pathlib import Path

import esm
import numpy as np
import torch
from Bio import SeqIO
from Bio.Align import substitution_matrices
from Bio.PDB import MMCIFParser, PDBParser
from Bio.SeqUtils import seq1
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import add_self_loops


# The order must match the original iFeatureOmegaCLI BLOSUM62 encoding.
# Reference: https://github.com/Superzchen/iFeatureOmega-CLI/blob/main/iFeatureOmegaCLI.py
AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"
DEFAULT_WEIGHTS = "/data/home/xiongwen2/software/GATSol/retrain_model/best_model.pt"
DEFAULT_ESM_MODEL = "/data/home/xiongwen2/software/esm1b_t33_650M_UR50S.pt"


class GATClassifier(nn.Module):
    """Preserve the original architecture and parameter names to load existing weights directly."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            GATConv(1300, 1024, heads=16),
            GATConv(1024 * 16, 1024, heads=16),
        ])
        self.lin1 = nn.Linear(1024 * 16, 128)
        self.lin2 = nn.Linear(128, 1)

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x = torch.relu(conv(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return self.lin2(torch.relu(self.lin1(x))).squeeze()


def read_sequence(sequence):
    """Accept a sequence string or the path to a single-record FASTA file."""
    text = "".join(sequence.split()).upper()
    if text and set(text) <= set(AMINO_ACIDS):
        sequence = text
    else:
        path = Path(sequence)
        if not path.is_file():
            raise ValueError("请输入标准氨基酸序列，或存在的单条 FASTA 文件路径。")
        records = list(SeqIO.parse(str(path), "fasta"))
        if len(records) != 1:
            raise ValueError("FASTA 文件必须恰好包含一条序列。")
        sequence = str(records[0].seq).upper()
    invalid = set(sequence) - set(AMINO_ACIDS)
    if invalid:
        raise ValueError(f"序列含非标准氨基酸：{', '.join(sorted(invalid))}")
    if not 1 <= len(sequence) <= 1022:
        raise ValueError("ESM-1b 要求序列长度为 1–1022；本脚本不截断序列。")
    return sequence


def read_coordinates(pdb_file, sequence, chain_id=None):
    """Use the first structure model and select the first chain that exactly matches the sequence."""
    path = Path(pdb_file)
    if not path.is_file():
        raise FileNotFoundError(f"找不到结构文件：{path}")
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    model = next(parser.get_structure("protein", str(path)).get_models())
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        residues = [r for r in chain if r.id[0] == " "]
        pdb_sequence = "".join(seq1(r.resname) for r in residues)
        if pdb_sequence == sequence:
            if any("CA" not in r for r in residues):
                raise ValueError(f"链 {chain.id!r} 缺少 CA 原子，无法建立完整接触图。")
            return np.asarray([r["CA"].coord for r in residues], dtype=np.float64)
    raise ValueError(
        "结构中没有与输入序列完全匹配的链。请检查序列、缺失残基和 --chain。"
    )


def make_edges(coordinates):
    """Follow the original workflow: CA distance < 10 Å, edges with i < j, then self-loops."""
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
    rows, cols = np.where(np.triu(distances < 10.0, k=1))
    edges = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    edges, _ = add_self_loops(edges, num_nodes=len(coordinates))
    return edges


@torch.no_grad()
def make_features(sequence, device, esm_model=None):
    """Combine 20-dimensional BLOSUM62 and 1280-dimensional ESM-1b layer 33 features per residue."""
    if esm_model:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(esm_model)
    else:
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model = model.to(device).eval()
    args = getattr(model, "args", None)
    if (getattr(args, "layers", None), getattr(args, "embed_dim", None),
            getattr(args, "arch", None)) != (33, 1280, "roberta_large"):
        raise ValueError("必须使用 esm1b_t33_650M_UR50S 权重，不能替换为 ESM-2。")
    _, _, tokens = alphabet.get_batch_converter()([("protein", sequence)])
    result = model(tokens.to(device), repr_layers=[33], return_contacts=False)
    embedding = result["representations"][33][0, 1:len(sequence) + 1].cpu()
    matrix = substitution_matrices.load("BLOSUM62")
    blosum = torch.tensor([
        [matrix[aa, other] for other in AMINO_ACIDS] for aa in sequence
    ], dtype=torch.float32)
    # Release the ESM model when this function returns, before loading GAT, to reduce peak memory usage.
    return torch.cat([blosum, embedding], dim=1)


def load_weights(model, weights):
    """Load the state_dict and handle GAT parameter renaming between PyG 2.3 and 2.5."""
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    expected = model.state_dict()
    for i in range(len(model.convs)):
        prefix = f"convs.{i}."
        if prefix + "lin.weight" in expected and prefix + "lin_src.weight" in state:
            source = state.pop(prefix + "lin_src.weight")
            target = state.pop(prefix + "lin_dst.weight", source)
            if not torch.equal(source, target):
                raise ValueError("权重的 GAT 源/目标变换不同，不能转换为当前网络。")
            state[prefix + "lin.weight"] = source
        elif prefix + "lin_src.weight" in expected and prefix + "lin.weight" in state:
            state[prefix + "lin_src.weight"] = state.pop(prefix + "lin.weight")
            if prefix + "lin_dst.weight" in expected:
                state[prefix + "lin_dst.weight"] = state[prefix + "lin_src.weight"]
    model.load_state_dict(state, strict=True)


def find_weights(weights=None):
    path = Path(weights or DEFAULT_WEIGHTS)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 GATSol 权重：{path}。可用 --weights 指定其他路径。")
    return path


@torch.no_grad()
def predict_sol(sequence, pdb_file, weights=None, chain=None, device="auto", esm_model=None):
    """Can also be called directly from Python; return sol as a float."""
    sequence = read_sequence(sequence)
    coordinates = read_coordinates(pdb_file, sequence, chain)
    weights = find_weights(weights)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前环境没有可用的 CUDA，请使用 --device cpu。")
    features = make_features(sequence, device, esm_model)
    graph = Data(
        x=features,
        edge_index=make_edges(coordinates),
        batch=torch.zeros(len(sequence), dtype=torch.long),
    ).to(device)
    model = GATClassifier()
    load_weights(model, weights)
    model = model.to(device).eval()
    return model(graph).item()


def main():
    # Allow older fair-esm versions to load a trusted local ESM-1b checkpoint.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    parser = argparse.ArgumentParser(description="预测 sol；不带参数时测试 CSV 第一条蛋白质。")
    parser.add_argument("sequence", nargs="?", help="氨基酸序列，或单条 FASTA 文件路径")
    parser.add_argument("pdb", nargs="?", help="PDB 结构文件路径，也支持 mmCIF")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help=f"GATSol 权重路径（默认：{DEFAULT_WEIGHTS}）")
    parser.add_argument("--chain", help="指定结构链，如 A；默认自动匹配序列")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--esm-model", default=DEFAULT_ESM_MODEL, help=f"本地 ESM-1b 权重路径（默认：{DEFAULT_ESM_MODEL}）")
    args = parser.parse_args()
    if (args.sequence is None) != (args.pdb is None):
        parser.error("sequence 和 pdb 请一起提供，或都不提供以运行默认测试。")
    try:
        if args.sequence is None:
            # Resolve test inputs relative to the script so direct execution finds the CSV and predicted structure.
            script_dir = Path(__file__).resolve().parent
            with (script_dir / "plp_sequences.csv").open(encoding="utf-8-sig", newline="") as handle:
                protein = next(csv.DictReader(handle))
            args.sequence = protein["sequence"]
            args.pdb = str(script_dir / "outputs" / "section2_protein_filtering" / "structures" / f"{protein['uniprot_id']}.pdb")
            print(f"Test protein: {protein['uniprot_id']} ({len(args.sequence)} residues)", flush=True)
            print(f"PDB: {args.pdb}", flush=True)
            print(f"GATSol weights: {args.weights}", flush=True)
            print(f"ESM-1b weights: {args.esm_model}", flush=True)
        sol = predict_sol(args.sequence, args.pdb, args.weights, args.chain, args.device, args.esm_model)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"错误：{error}\n")
    print(f"sol = {sol:.6f}")


if __name__ == "__main__":
    main()

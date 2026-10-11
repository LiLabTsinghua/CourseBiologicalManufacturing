"""Predict kcat in batches: predict_kcat(sequences, smiles) returns kcat (s^-1) in input order."""

from functools import lru_cache
from pathlib import Path
import pickle
import re

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

if __package__:
    from . import build_vocab
    from .pretrain_trfm import TrfmSeq2seq
    from .utils import split
else:
    import build_vocab
    from pretrain_trfm import TrfmSeq2seq
    from utils import split

MODEL_DIR = Path(__file__).resolve().parent
# The protein language model uses a shared server directory; other weights are in this script's directory.
PROT_T5_DIR = Path("/data/home/xiongwen2/software/prot_t5_xl_uniref50")


# The original vocabulary was saved as __main__.WordVocab; support that name when loading via imports.
class _VocabUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ("__main__", "build_vocab") and name in ("WordVocab", "Vocab", "TorchVocab"):
            return getattr(build_vocab, name)
        return super().find_class(module, name)


@lru_cache(maxsize=1)
def _load_models():
    """Load models on the first call and reuse them on subsequent calls."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with (MODEL_DIR / "vocab.pkl").open("rb") as f:
        vocab = _VocabUnpickler(f).load()
    smiles_model = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    smiles_model.load_state_dict(torch.load(str(MODEL_DIR / "trfm_12_23000.pkl"), map_location="cpu"))
    smiles_model.eval()
    protein_dir = str(PROT_T5_DIR)
    tokenizer = T5Tokenizer.from_pretrained(protein_dir, do_lower_case=False, local_files_only=True)
    protein_model = T5EncoderModel.from_pretrained(protein_dir, local_files_only=True).to(device).eval()
    with (MODEL_DIR / "UniKP for kcat.pkl").open("rb") as f:
        kcat_model = pickle.load(f)
    return vocab, smiles_model, tokenizer, protein_model, kcat_model, device


@torch.no_grad()
def _smiles_to_vec(smiles, vocab, model):
    inputs = []
    for sm in smiles:
        tokens = split(sm).split()
        if len(tokens) > 218:
            tokens = tokens[:109] + tokens[-109:]
        ids = [3] + [vocab.stoi.get(token, 1) for token in tokens] + [2]
        inputs.append(ids + [0] * (220 - len(ids)))
    return model.encode(torch.tensor(inputs).t())


@torch.no_grad()
def _sequences_to_vec(sequences, tokenizer, model, device):
    features = []
    # Extract protein features one sequence at a time to limit GPU memory usage for long sequences.
    for seq in sequences:
        if len(seq) > 1000:
            seq = seq[:500] + seq[-500:]
        seq = " ".join(re.sub(r"[UZOB]", "X", seq))
        ids = tokenizer.batch_encode_plus([seq], add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids["input_ids"]).to(device)
        mask = torch.tensor(ids["attention_mask"]).to(device)
        embedding = model(input_ids=input_ids, attention_mask=mask).last_hidden_state[0].cpu().numpy()
        length = int(mask.sum().item()) - 1  # Exclude the final special token.
        features.append(embedding[:length].mean(axis=0, dtype=np.float64))
    return np.asarray(features)


def predict_kcat(sequences, smiles):
    """Pair sequences[i] with smiles[i] and return the corresponding list of kcat values."""
    if isinstance(sequences, str) or isinstance(smiles, str):
        raise TypeError("请传入序列列表和 SMILES 列表，而不是单个字符串")
    if len(sequences) != len(smiles):
        raise ValueError("序列和 SMILES 的数量必须相同")
    if len(sequences) == 0:
        return []
    sequences = [re.sub(r"\s+", "", seq).upper() for seq in sequences]
    smiles = [sm.strip() for sm in smiles]
    if any(not seq or not sm for seq, sm in zip(sequences, smiles)):
        raise ValueError("序列和 SMILES 不能为空")
    if any(not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYUXZOB]+", seq) for seq in sequences):
        raise ValueError("序列中含有无效的氨基酸字符")

    vocab, smiles_model, tokenizer, protein_model, kcat_model, device = _load_models()
    smiles_vec = _smiles_to_vec(smiles, vocab, smiles_model)
    seq_vec = _sequences_to_vec(sequences, tokenizer, protein_model, device)
    features = np.concatenate((smiles_vec, seq_vec), axis=1)
    return np.power(10.0, np.asarray(kcat_model.predict(features), dtype=float)).tolist()

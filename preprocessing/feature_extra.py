import torch
import esm
import os
os.environ['TORCH_HOME']='~/workspace/'
import numpy as np
import pandas as pd
import lmdb
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import pickle
import glob
from multiprocessing import Pool
from collections import defaultdict
from Bio import PDB
from torch import nn as nn
from unimol_tools import UniMolRepr


def run_esm(enzyme_seq, batch_size):
    torch.cuda.empty_cache()
    # Load ESM-2 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)

    model = nn.DataParallel(model)
    
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        (f"protein{i}", enzyme_seq[i]) for i in range(len(enzyme_seq))
    ]
    while len(data) < batch_size:
        data.append(data[0])
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        torch.cuda.empty_cache()

    token_representations = results["representations"][33].cpu()

    del results
    del batch_tokens
    return token_representations

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    # sequence_representations = []
    # for i, tokens_len in enumerate(batch_lens):
    #     sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    # Look at the unsupervised self-attention map contact predictions
    # import matplotlib.pyplot as plt
    # for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
    #     plt.matshow(attention_contacts[: tokens_len, : tokens_len])
    #     plt.title(seq)
    #     plt.show()

def run_esm_fold(sequence):
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()

    # Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
    # Lower sizes will have lower memory requirements at the cost of increased speed.
    # model.set_chunk_size(128)

    # Multimer prediction can be done with chains separated by ':'

    with torch.no_grad():
        output = model.infer_pdb(sequence)

    with open("result.pdb", "w") as f:
        f.write(output)

    return

    # import biotite.structure.io as bsio
    # struct = bsio.load_structure("result.pdb", extra_fields=["b_factor"])
    # print(struct.b_factor.mean())  # this will be the pLDDT
    # # 88.3


def run_esm_if(path='/root/workspace/enzyme/MetaEnzyme/preprocessing/'):
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()

    fpath = path + 'result.pdb' # .pdb format is also acceptable

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(file=fpath,id=None)
    chains = structure.get_chains()
    chain_ids = [chain.id for chain in chains]
    reps = []
    for chain_id in chain_ids:
        structure = esm.inverse_folding.util.load_structure(fpath, chain_id)
        coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
        # print('Native sequence:')
        # print(native_seq)

        reps.append(esm.inverse_folding.util.get_encoder_output(model, alphabet, coords))
        # print(len(coords))
        # print(rep.shape)
        # print(rep)
    return reps


def run_unimol(smiles_list):
    # single smiles unimol representation
    clf = UniMolRepr(data_type='molecule', remove_hs=False)
    unimol_repr = clf.get_repr(smiles_list, return_atomic_reprs=True)
    # CLS token repr
    print(np.array(unimol_repr['cls_repr']).shape)
    # atomic level repr, align with rdkit mol.GetAtoms()
    print(np.array(unimol_repr['atomic_reprs']).shape)
    return 0

run_unimol(['c1ccc(cc1)C2=NCC(=O)Nc3c2cc(cc3)[N+](=O)[O]'])
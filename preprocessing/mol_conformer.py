import os
import sys
sys.path.append("../")
from utils import *
import numpy as np
from tqdm import tqdm
from unimol_tools.data import DataHub
import pickle


SAVE_PATH = '/root/workspace/enzyme/MetaEnzyme/dataset/mol_conformers'
MOL_PATH = '/root/workspace/enzyme/MetaEnzyme/dataset/smile'

total_mol_names = [item for item in os.listdir(MOL_PATH) if item.endswith('.smi')]
mol_nums = len(total_mol_names)
batch_size = 32
epoch = mol_nums // batch_size + 1
params = {'data_type': 'molecule', 'remove_hs': False}

for i in tqdm(range(epoch)):
    batch_mol_names = total_mol_names[i*batch_size: (i+1)*batch_size]
    mol_seqs = []
    mol_names = []
    for item in batch_mol_names:
        with open(os.path.join(MOL_PATH, item)) as smi_file:
            seq, name = smi_file.readline().split(" ")
            mol_seqs.append(seq)
            mol_names.append(name)
    mol_process_seqs =  np.array(mol_seqs)
    datahub = DataHub(data=mol_process_seqs, 
                    task='repr', 
                    is_train=False,
                    **params,
                )
    mol_smiles = datahub.data['smiles']
    mol_process_seqs = datahub.data['unimol_input']

    for mol_seq, mol_name, mol_smile, mol_process_seq in zip(mol_seqs, mol_names, mol_smiles, mol_process_seqs):
        assert mol_seq == mol_smile
        if not os.path.exists(os.path.join(SAVE_PATH, f"{mol_name}.pkl")):
            with open(os.path.join(SAVE_PATH, f"{mol_name}.pkl"), "wb") as file:
                pickle.dump(mol_process_seq, file)

    
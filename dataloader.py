import os
os.environ['TORCH_HOME']='~/workspace/'
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random
import esm
from unimol_tools.data import DataHub
import numpy as np
import torch
from utils import *
import pickle


DATA_DICT = ['esp', 'MPEK', 'KM', 'DLKcat', 'HXKM']
# 输入参数决定训练的任务对应需要加载的数据集，需要：1.处理数据csv文件，2.dataloader
# 对比学习需不需要筛选掉重复的序列，比如一个batch中有多个数据是同一个氨基酸序列

class EspDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.mol_conformers_path = '/root/workspace/enzyme/MetaEnzyme/dataset/mol_conformers'

    def __getitem__(self, index):
        v_smile_ID = self.data.iloc[index,:]['mol_ID']
        v_protein = self.data.iloc[index,:]['Protein']
        label = self.data.iloc[index,:]['Y']

        assert os.path.exists(os.path.join(self.mol_conformers_path, f"{v_smile_ID}.pkl")), f"Molecular {v_smile_ID} comformer does not exist."

        with open(os.path.join(self.mol_conformers_path, f"{v_smile_ID}.pkl"), "rb") as file:
            v_smile = pickle.load(file)


        if len(v_protein) >= 1024:
            start_index = random.randint(0, len(v_protein) - 1024)
            v_protein = v_protein[start_index: start_index + 1024]

        return v_smile, v_protein, label
    
    def __len__(self):
        return len(self.data)

class MPEKDataset(Dataset):
    pass

class KMDataset(Dataset):
    pass

class DLKcatDataset(Dataset):
    pass

class HXKMDataset(Dataset):
    pass

class MultiDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.mol_conformers_path = '/root/workspace/enzyme/MetaEnzyme/dataset/mol_conformers'

    def __getitem__(self, index):
        v_smile_ID = self.data.iloc[index,:]['mol_ID']

        assert os.path.exists(os.path.join(self.mol_conformers_path, f"{v_smile_ID}.pkl")), f"Molecular {v_smile_ID} comformer does not exist."

        with open(os.path.join(self.mol_conformers_path, f"{v_smile_ID}.pkl"), "rb") as file:
            v_smile = pickle.load(file)


        v_protein = self.data.iloc[index,:]['Protein']

        if len(v_protein) >= 1024:
            start_index = random.randint(0, len(v_protein) - 1024)
            v_protein = v_protein[start_index: start_index + 1024]

        return v_smile, v_protein
    
    def __len__(self):
        return len(self.data)


class Data_reader():
    def __init__(self, data_path, data_name, task):
        ### data_path: /root/workspace/enzyme/MetaEnzyme/dataset
        if data_name not in DATA_DICT:
            raise NotImplementedError
        self.task = task
        self.data_name = data_name
        self.data_path = data_path
        self.data_reader = {
            'esp': self.read_esp_data,
            'MPEK': self.read_mpek_data,
            'KM': self.read_km_data,
            'DLKcat': self.read_DLKcat_data,
            'HXKM': self.read_hxkm_data,
        }

    def read_data(self):
        if self.task == 'comparative':
            return self.read_multi_data()
        else:
            return self.data_reader[self.data_name]()

    def read_esp_data(self):
        # train test validation
        files = {
            'train': os.path.join(self.data_path, "esp", "modified_train.csv"),
            'test': os.path.join(self.data_path, "esp", "modified_test.csv"),
            'validation': os.path.join(self.data_path, "esp", "modified_val.csv")
        }
        dfs = {key: pd.read_csv(path).loc[:, ['Protein', 'mol_ID', "Y"]] for key, path in files.items()}
        return EspDataset(data=dfs['train']),  EspDataset(data=dfs['validation']), EspDataset(data=dfs['test'])

    def read_mpek_data(self):
        raise NotImplementedError

    def read_km_data(self):
        raise NotImplementedError
    
    def read_DLKcat_data(self):
        raise NotImplementedError

    def read_hxkm_data(self):
        raise NotImplementedError
    
    def read_multi_data(self):
        dfs = []
        for data_name in DATA_DICT:
            for file in os.listdir(os.path.join(self.data_path, data_name)):
                if 'modified_train' in file and data_name != "esp":
                    dfs.append(pd.read_csv(os.path.join(self.data_path, data_name, file)).loc[:,['Protein','mol_ID']])
                elif 'modified_train' in file and data_name == "esp":
                    dfs.append(pd.read_csv(os.path.join(self.data_path, data_name, file)).loc[lambda df: df['Y'] == 1, ['Protein','mol_ID']])
        dfs = pd.concat(dfs)
        return MultiDataset(data=dfs)

esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
del esm_model

protein_batch_converter = esm_alphabet.get_batch_converter()

def collate_fn(data):
    mol_seq = []
    pro_seq = []
    labels = []

    for item in data:
        if len(item) == 3:
            labels.append(torch.tensor(item[2]))
        mol_seq.append(item[0])
        pro_seq.append(item[1])
    
    mol_rep = mol_preprocessing(mol_seq)

    pro_rep, pro_mask = protein_preprocessing(pro_seq, protein_batch_converter)

    if len(labels) != 0:
        assert len(labels) == len(pro_rep)
        labels = torch.stack(labels)
    return (mol_rep, pro_rep, pro_mask, labels)
    # return(mol_seq, pro_seq)

def protein_preprocessing(pro_seq, protein_batch_converter):
    _, _, batch_tokens = protein_batch_converter([(f"protein{i}", pro_seq[i]) for i in range(len(pro_seq))])

    batch_lens = (batch_tokens != esm_alphabet.padding_idx).sum(1)
    padding_masks = []
    for i, token_len in enumerate(batch_lens):
        padding_mask = torch.zeros([batch_tokens.shape[1]], dtype=torch.bool, device=batch_tokens.device)
        padding_mask[token_len + 1:] = True
        padding_masks.append(padding_mask)
    padding_masks = torch.stack(padding_masks)
    return batch_tokens, padding_masks
    # return 0

def mol_preprocessing(mol_seq):
    # params = {'data_type': 'molecule', 'remove_hs': False}

    # mol_seq = np.array(mol_seq)
    # datahub = DataHub(data=mol_seq, 
    #                 task='repr', 
    #                 is_train=False, 
    #                 **params,
    #             )
    # mol_seq = datahub.data['unimol_input']

    mol_seq, label = mol_batch_collate_fn(mol_seq)

    net_input, _ = decorate_torch_batch((mol_seq, label))

    return net_input
    # return mol_seq
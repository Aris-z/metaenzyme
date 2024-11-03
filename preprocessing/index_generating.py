import os
import pandas as pd
import numpy
import random
import string


# 去重并在对应文件夹中生成命名后的蛋白质序列文件以及小分子SMILE文件，同时在每个源数据csv中生成对应的ID

DATASET_PATH = '/root/workspace/enzyme/MetaEnzyme/dataset/'
DATASET = [
    '/root/workspace/enzyme/MetaEnzyme/dataset/DLKcat',
    '/root/workspace/enzyme/MetaEnzyme/dataset/esp',
    '/root/workspace/enzyme/MetaEnzyme/dataset/HXKM',
    '/root/workspace/enzyme/MetaEnzyme/dataset/KM',
    '/root/workspace/enzyme/MetaEnzyme/dataset/MPEK'
]


def generate_unique_labels(data_set):
    data_set = list(data_set)
    labels = set()
    result = {}
    
    while len(result) < len(data_set):
        # 生成一个长度为4的随机标签
        label = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
        while label in labels:
            label = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
        item = data_set[len(result)]
        result[item] = label
        labels.add(label)
    
    return result

unique_protein_sequences = set()
unique_mol_sequences = set()

# 第一步：读取所有 CSV 文件并收集唯一的序列
for dataset_path in DATASET:
    for filename in os.listdir(dataset_path):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(dataset_path, filename))
            for seq in df['Protein'].unique():
                if seq not in unique_protein_sequences:
                    unique_protein_sequences.add(seq)
            for seq in df['SMILES'].unique():
                if seq not in unique_mol_sequences:
                    unique_mol_sequences.add(seq)

unique_protein_sequences = generate_unique_labels(unique_protein_sequences)
unique_mol_sequences = generate_unique_labels(unique_mol_sequences)

# 第二步：处理每个 CSV 文件，添加序列名称
for dataset_path in DATASET:
    for filename in os.listdir(dataset_path):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(dataset_path, filename))
            
            # 添加全局序列名称
            df['protein_ID'] = df['Protein'].map(unique_protein_sequences)
            df['mol_ID'] = df['SMILES'].map(unique_mol_sequences)

            # 保存修改后的数据集为新的 CSV 文件
            modified_filename = f"modified_{filename}"
            df.to_csv(os.path.join(dataset_path, modified_filename), index=False)

# 创建 FASTA 文件
for seq, name in unique_protein_sequences.items():
    fasta_file_name = f"{name}.fasta"
    if not os.path.exists(os.path.join(DATASET_PATH + "fasta", fasta_file_name)):
        with open(os.path.join(DATASET_PATH + "fasta", fasta_file_name), 'w') as fasta_file:
            fasta_file.write(f">{name}\n")
            fasta_file.write(f"{seq}\n")

for seq, name in unique_mol_sequences.items():
    fasta_file_name = f"{name}.smi"
    if not os.path.exists(os.path.join(DATASET_PATH + "smile", fasta_file_name)):
        with open(os.path.join(DATASET_PATH + "smile", fasta_file_name), 'w') as fasta_file:
            fasta_file.write(f"{seq} {name}")

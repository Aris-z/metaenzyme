import os
import numpy
import pandas as pd
from tqdm import tqdm
import pickle
import argparse
import random
from dataloader import Data_reader, collate_fn
from torch.utils.data import DataLoader
from model import MetaEnzyme
from trainer import Trainer


def run(data_path, args, feature_type, data_name):
    if feature_type not in ['enzyme_2D', 'enzyme_3D', 'mol']:
        raise NotImplementedError

    data_reader = Data_reader(data_path, data_name, args.task)
    model = MetaEnzyme(args.task, args.data_name)
    trainer = Trainer(args)

    if args.task == 'comparative':
        train_dataset = data_reader.read_data()
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=False)
        
        global_step, avg_loss = trainer.pretrain(train_dataloader, model)
        print(f"Training done: total_step = {global_step}, avg loss = {avg_loss}")

    elif args.task == 'mlp':
        train_dataset, val_dataset, test_dataset = data_reader.read_data()

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=False)
        validation_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=False)
        test_dataloadeer = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=False)
        global_step, avg_loss = trainer.train(train_dataloader, validation_dataloader, test_dataloadeer, model)
        print("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)

    
    return


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--file_name', type=str, default='')
    parser.add_argument('--data_name', type=str, default='esp')
    parser.add_argument('--feature_type', type=str, default='enzyme_2D')
    parser.add_argument('--task', type=str, choices=['comparative', 'mlp', 'infer'], default='comparative')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--checkpoint', type=str, default='/root/workspace/enzyme/MetaEnzyme/ckpt/comparative_checkpoint_1_400.pt')


    args = parser.parse_args() 

    # feature_type = 'enzyme_3D'
    # feature_type = 'mol'
    data_path = '/root/workspace/enzyme/MetaEnzyme/dataset'
    run(data_path, args, args.feature_type, data_name=args.data_name)

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import torch.nn.functional as F
import os
from molebert import GNN, GNN_graphpred



import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class Trainer(object):
    def __init__(self, args):
        self.lr = args.lr
        self.weight_decay = 0.01
        self.eps = 1e-8
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.logging_steps = 1
        self.num_train_epochs = 2
        self.save_steps = 100

        self.task = args.task
        self.ckpt = args.checkpoint
        self.eval_steps = 8

    def pretrain(self, train_dataloader, model):
        '''
        Trains the model.
        '''

        # total training iterations
        # t_total = len(train_dataloader) // config.gradient_accumulation_steps \
        #             * config.num_train_epochs
        model_1 = GNN_graphpred(5, 300, 1, JK = "last", drop_ratio = 0.5, graph_pooling = "mean", gnn_type = "gin")
        model_1.from_pretrained('/root/workspace/enzyme/MetaEnzyme/Mole-BERT.pth')
        model_1.to("cuda:0")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:        # 判断GPU个数
            model = nn.DataParallel(model) 
        
        model = model.to(device)
        # model.load_state_dict(torch.load('/root/workspace/enzyme/MetaEnzyme/ckpt/comparative_checkpoint_1_406__1104_model_0.pt'))
        # for param in model.parameters():
        #     param.requires_grad = False
        model.train()

        params = [
            {"params": model_1.parameters(), "lr": 1e-4},
            {"params": model.parameters(), "lr": 5e-5}
        ]

        optimizer = AdamW(params, lr=self.lr, eps=self.eps, weight_decay=self.weight_decay)
        # optimizer = SGD(model.parameters(), lr=self.lr, momentum=0.9)


        # Warmup iterations = 20% of total iterations
        # num_warmup_steps = int(0.20 * t_total)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,
            T_mult=2,
            eta_min=1e-8
        )

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataloader))
        # logger.info("  Num Epochs = %d", config.num_train_epochs)
        logger.info("  Number of GPUs = %d", torch.cuda.device_count())

        # logger.info("  Batch size per GPU = %d", config.per_gpu_train_batch_size)
        # logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
        #             config.train_batch_size * config.gradient_accumulation_steps)
        # logger.info("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)
        # logger.info("  Total optimization steps = %d", t_total)


        global_step, global_loss, global_acc = 0, 0.0, 0.0
        model_1.zero_grad()
        model.zero_grad()

        for epoch in range(self.num_train_epochs):
            for step, batch in enumerate(tqdm(train_dataloader)):
                torch.cuda.empty_cache()
                input_mol, input_protein, protein_mask, _ = batch

                # test = torch.rand(32,1).to(device)

                # # 这个地方应该要先把序列表示成特征才能放到cuda里面去
                # for k in input_mol.keys():
                #     input_mol[k] = input_mol[k].to(device)
                input_mol = input_mol.to(device)

                input_protein = input_protein.to(device)
                protein_mask = protein_mask.to(device)

                _, input_mol = model_1(input_mol.x, input_mol.edge_index, input_mol.edge_attr, input_mol.batch)

                # 特征计算
                mol_features, protein_features = model(input_mol, input_protein, protein_mask)

                # if config.n_gpu == 1:
                #     logit_scale = model.logit_scale.exp()
                # elif config.n_gpu > 1:
                #     logit_scale = model.module.logit_scale.exp()
                logit_scale = model.module.logit_scale.exp()

                logits_per_mol = logit_scale * mol_features @ protein_features.t()
                logits_per_protein = logit_scale * protein_features @ mol_features.t()

                labels = torch.arange(len(logits_per_mol)).to(logits_per_mol.device)

                mol_loss = F.cross_entropy(logits_per_mol, labels)
                protein_loss  = F.cross_entropy(logits_per_protein, labels)

                loss = (mol_loss + protein_loss) / 2

                # if config.n_gpu > 1: 
                #     loss = loss.mean() # mean() to average on multi-gpu parallel training
                # if config.gradient_accumulation_steps > 1:
                #     loss = loss / config.gradient_accumulation_steps
                loss = loss.mean()
                loss = loss / self.gradient_accumulation_steps

                loss.backward()

                global_loss += loss.item()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    global_step += 1
                    optimizer.step() # PYTORCH 1.x : call optimizer.step() first then scheduler.step()
                    
                    # logit scaling set as max 100 as mentioned in CLIP paper # log(100) = 4.6052
                    # if config.n_gpu == 1:
                    #     model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
                    # elif config.n_gpu > 1:
                    #     model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)
                    model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)

                    if scheduler:
                        scheduler.step() 

                    model_1.zero_grad()
                    model.zero_grad()

                    if global_step % self.logging_steps == 0:
                        logger.info("Epoch: {}, global_step: {}, lr: {}, loss: {} ({})".format(epoch, global_step, 
                            optimizer.param_groups[0]["lr"], loss.item(), global_loss / global_step)
                        )

                    if (self.save_steps > 0 and global_step % self.save_steps == 0):
                        # saving checkpoint
                        save_checkpoint(epoch, global_step, model, model_1, self.task)
            logger.info("End of Epoch: {}".format(epoch))
        save_checkpoint(epoch, global_step, model, model_1, self.task)

        return global_step, global_loss / global_step
    

    def train(self, train_dataloader, validation_dataloader, test_dataloader, model):
        '''
        Trains the model.
        '''
        self.model_1 = GNN_graphpred(5, 300, 1, JK = "last", drop_ratio = 0.5, graph_pooling = "mean", gnn_type = "gin")
        # self.model_1.from_pretrained('/root/workspace/enzyme/MetaEnzyme/ckpt/comparative_checkpoint_1_400_model_1.pt')
        self.model_1.load_state_dict(torch.load('/root/workspace/enzyme/MetaEnzyme/ckpt/comparative_checkpoint_1_400_model_1.pt'))
        for param in self.model_1.parameters():
            param.requires_grad = False
        self.model_1.to("cuda:0")

        self.model = model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:        # 判断GPU个数
            self.model = nn.DataParallel(self.model) 
        
        self.model = self.model.to(self.device)

        assert os.path.exists(self.ckpt)

        self.model.load_state_dict(torch.load(self.ckpt), strict=False)
        # for name, param in model.named_parameters():
        #     if "downhead" not in name:
        #         param.requires_grad = False
        for name, param in self.model.named_parameters():
            if "downhead"  not in name:
                param.requires_grad = False

        self.model.train()

        params = [
            {'params': [p for n, p in self.model.named_parameters() if "downhead" in n], 'lr': 3e-3},
            # {'params': [p for n, p in self.model.named_parameters() if "downhead" not in n], 'lr': self.lr*0.1},
            # {"params": self.model_1.parameters(), "lr": self.lr*0.1}
        ]
        # optimizer = AdamW(self.model.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.weight_decay)
        optimizer = AdamW(params, lr=self.lr, eps=self.eps, weight_decay=self.weight_decay)

        loss_func = nn.BCEWithLogitsLoss()

        # Warmup iterations = 20% of total iterations
        # num_warmup_steps = int(0.20 * t_total)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,
            T_mult=2,
            eta_min=1e-8
        )

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataloader))
        # logger.info("  Num Epochs = %d", config.num_train_epochs)
        logger.info("  Number of GPUs = %d", torch.cuda.device_count())


        # logger.info("  Batch size per GPU = %d", config.per_gpu_train_batch_size)
        # logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
        #             config.train_batch_size * config.gradient_accumulation_steps)
        # logger.info("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)
        # logger.info("  Total optimization steps = %d", t_total)


        global_step, global_loss, global_acc = 0, 0.0, 0.0
        self.model.zero_grad()
        self.model_1.zero_grad()

        for epoch in range(self.num_train_epochs):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
                torch.cuda.empty_cache()
                input_mol, input_protein, protein_mask, label = batch
                # print(input_mol.shape)

                # # 这个地方应该要先把序列表示成特征才能放到cuda里面去
                # for k in input_mol.keys():
                #     input_mol[k] = input_mol[k].to(self.device)
                input_mol = input_mol.to(self.device)

                input_protein = input_protein.to(self.device)
                label = label.to(self.device)
                protein_mask = protein_mask.to(self.device)

                _, input_mol = self.model_1(input_mol.x, input_mol.edge_index, input_mol.edge_attr, input_mol.batch)

                # 特征计算
                output = self.model(input_mol, input_protein, protein_mask)
                # normalized features
                # mol_features = mol_features / mol_features.norm(dim=-1, keepdim=True)
                # protein_features = protein_features / protein_features.norm(dim=-1, keepdim=True)
                # mol_features = F.normalize(mol_features, dim=-1)
                # protein_features = F.normalize(protein_features, dim=-1)

                # loss = F.cross_entropy(output, label)
                label = F.one_hot(label, num_classes=2).to(torch.float)
                loss = loss_func(output, label)

                loss = loss.mean()
                loss = loss / self.gradient_accumulation_steps
                
                loss.backward()

                global_loss += loss.item()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    global_step += 1
                    optimizer.step() # PYTORCH 1.x : call optimizer.step() first then scheduler.step()

                    if scheduler:
                        scheduler.step() 
                        
                    self.model.zero_grad()
                    self.model_1.zero_grad()

                    if global_step % self.eval_steps == 0:
                        acc, eval_loss = self.eval(validation_dataloader)
                        logger.info("eval loss: {}, acc: {}".format(eval_loss, acc)
                        )

                    if global_step % self.logging_steps == 0:
                        logger.info("Epoch: {}, global_step: {}, lr: {}, loss: {} ({})".format(epoch, global_step, 
                            optimizer.param_groups[0]["lr"], loss.item(), global_loss / global_step)
                        )

                    if (self.save_steps > 0 and global_step % self.save_steps == 0):
                        # saving checkpoint
                        save_checkpoint(epoch, global_step, self.model, self.model_1, self.task)

            logger.info("End of Epoch: {}".format(epoch))
        save_checkpoint(epoch, global_step, self.model, self.model_1, self.task)

        return global_step, global_loss / global_step
    
    def eval(self, val_dataloader):
        torch.cuda.empty_cache()
        global_step, global_loss, correct = 0, 0.0, 0.0
        with torch.no_grad():
            for _, batch in enumerate(tqdm(val_dataloader, desc='Evaluation')):
                global_step += 1
                input_mol, input_protein, protein_mask, label = batch
                # for k in input_mol.keys():
                #     input_mol[k] = input_mol[k].to(self.device)
                input_mol = input_mol.to(self.device)
                input_protein = input_protein.to(self.device)
                label = label.to(self.device)
                protein_mask = protein_mask.to(self.device)         

                # print(input_mol.device)
                _, input_mol = self.model_1(input_mol.x, input_mol.edge_index, input_mol.edge_attr, input_mol.batch)

                # 特征计算
                output = self.model(input_mol, input_protein, protein_mask)

                correct += (torch.argmax(F.sigmoid(output), dim=1) == label).sum().item() / len(label)

                loss = F.cross_entropy(output, label)
                loss = loss.mean()

                global_loss += loss.item()

            global_loss = global_loss / global_step
            acc = correct / global_step

        return acc, global_loss
    

def save_checkpoint(epoch, global_step, model, model_1, task):
    checkpoint_path = '/root/workspace/enzyme/MetaEnzyme/ckpt'
    checkpoint_path_0 = os.path.join(checkpoint_path, f'{task}_checkpoint_{epoch}_{global_step}_model_0.pt')
    checkpoint_path_1 = os.path.join(checkpoint_path, f'{task}_checkpoint_{epoch}_{global_step}_model_1.pt')
    save_num = 0
    # while (save_num < 10):
        # try:
    torch.save(model.state_dict(), checkpoint_path_0)
    torch.save(model_1.state_dict(), checkpoint_path_1)
    print("Save checkpoint to {}".format(checkpoint_path))
            # break
        # except:
        #     save_num += 1
    # if save_num == 10:
    #     logger.info("Failed to save checkpoint after 10 trails.")
    return

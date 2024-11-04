import os
os.environ['TORCH_HOME']='~/workspace/'
import torch
from torch import nn
import esm
from unimol_tools.models import UniMolModel
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from molebert import GNN, GNN_graphpred


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        self.attn_mask = attn_mask.to(dtype=x.dtype, device=x.device) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=attn_mask)[0]

    def forward(self, input):
        if type(input) == tuple:
            x, attn_mask = input
        elif type(input) == torch.Tensor:
            x = input
            attn_mask = None
        else:
            print(type(input))
            raise TypeError
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor=None):
        return self.resblocks((x, attn_mask))



class MetaEnzyme(nn.Module):
    def __init__(self, task, data_name):
        super(MetaEnzyme, self).__init__()
        self.esm_model, _ = esm.pretrained.esm2_t33_650M_UR50D()
        for param in self.esm_model.parameters():
            param.requires_grad = False

        # self.unimol_model = UniMolModel(output_dim=1, data_type='molecule', remove_hs=False)
        # for param in self.unimol_model.parameters():
        #     param.requires_grad = False
        # self.unimol_model = GNN_graphpred(5, 300, 1, JK = "last", drop_ratio = 0.5, graph_pooling = "mean", gnn_type = "gin")
        # self.unimol_model.from_pretrained('/root/workspace/enzyme/MetaEnzyme/Mole-BERT.pth')

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        #padding的时候是否需要mask
        # self.mol_project_layer = Transformer(512, 8, 8)
        self.mol_project_layer = nn.Sequential(OrderedDict([
                  ('unimol_mlp1', nn.Linear(300, 1024)),
                  ('gelu1', nn.GELU()),
                  ('unimol_mlp2', nn.Linear(1024, 512)),
                  ('gelu2', nn.GELU()),
                  ('unimol_mlp3', nn.Linear(512, 256)),
                  ('gelu2', nn.GELU()),
                ]))

        self.protein_project_layer = Transformer(1280, 8, 8)

        self.task = task
        self.data_name = data_name

        if self.task == 'mlp':
            if data_name in ['esp']:
                self.l1_downhead = nn.Linear(512, 1024)
                self.l2_downhead = nn.Linear(1024, 256)
                self.class_gelu = nn.GELU()
                self.classfication_downhead = nn.Linear(256, 2)
            else:
                raise NotImplementedError

        self.protein_mlp = nn.Linear(1280, 256)

        # self.MLP_layer = nn.Linear()

    def forward(self, mol_rep, protein_rep, protein_mask):
        ## mask屏蔽padding
        protein_rep = self.embedding(mol_rep, protein_rep)

        # device = test.device
        # mol_rep = torch.rand(4,512,512).to(device)
        # protein_rep = torch.rand(4,512,1280).to(device)
        #####################
        mol_rep = self.mol_project_layer(mol_rep)

        protein_rep, _ = self.protein_project_layer(protein_rep, protein_mask)
        seq_lens = (protein_mask == False).sum(1)
        protein_seq_represents = []
        for i, token_len in enumerate(seq_lens):
            protein_seq_represents.append(protein_rep[i, 1:token_len+1, :].mean(0))
        protein_rep = self.protein_mlp(torch.stack(protein_seq_represents))

        ####################
        # protein_rep = torch.randint(0, 10, protein_rep.shape, dtype=protein_rep.dtype, device=protein_rep.device)
        # mol_rep = torch.randint(0, 10, mol_rep.shape, dtype=mol_rep.dtype, device=mol_rep.device)

        ## pooling对齐特征
        ################
        # logits_per_mol = mol_rep @ protein_rep.t()
        # logits_per_protein = logits_per_mol.t()
        ################
        if self.task == 'comparative':
            return mol_rep, protein_rep
        elif self.data_name in ['esp']:
            try:
                assert self.classfication_downhead != None
            except:
                raise TypeError
            mlp_input = torch.concat((mol_rep, protein_rep), dim=1)
            output = self.classfication_downhead(self.class_gelu(self.l2_downhead(self.class_gelu(self.l1_downhead(mlp_input)))))
            return output
        else:
            raise NotImplementedError

        # # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()

        # # shape = [global_batch_size, global_batch_size]
        # return logits_per_image, logits_per_text

    def embedding(self, mol_seq, pro_seq):
        # batch_tokens = batch_tokens.to(device)
        with torch.no_grad():            
            results = self.esm_model(pro_seq, repr_layers=[33], return_contacts=False)

        protein_rep = results["representations"][33].detach()
        del results

        # # mol_rep = mol_rep.detach().cpu()
        # mol_rep = torch.rand(protein_rep.shape[0], 512, 512)
        # protein_rep = protein_rep.detach().cpu()

        return protein_rep
    
    
    # def pad_to_shape(self, tensor, shape=1026):
    #     pad_height = shape - tensor.shape[-2]
    #     # 使用 F.pad 填充，模式为 constant
    #     tensor = F.pad(tensor, (0, 0, 0, pad_height), mode='constant', value=0)
    #     return tensor
    def pad_to_shape(self, tensor, shape=1026):
        pad_height = shape - tensor.shape[-2]
        padding_tensor = F.pad(tensor, (0, 0, 0, pad_height), mode='constant', value=1)
        if len(tensor.shape) == 2: #[seq, fea]
            padding_mask = torch.zeros(shape, dtype=torch.bool, device=tensor.device)
            padding_mask[tensor.shape[0]:] = True
        else:
            raise TypeError
        # 使用 F.pad 填充，模式为 constant
        return padding_tensor, padding_mask
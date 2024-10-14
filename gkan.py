# Importar bibliotecas necessárias para o projeto 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gc

# Importar bibliotecas geométricas PyTorch 
import torch_geometric.transforms as T
from torch_geometric.utils import *
from torch_geometric.datasets import Planetoid

class GKAN(torch.nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, grid_feat, num_layers, use_bias=False):
        super().__init__()
        self.num_layers = num_layers
        self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.lins = torch.nn.ModuleList()
        for i in range(num_layers):
            self.lins.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))
        self.lins.append(nn.Linear(hidden_feat, out_feat, bias=False))

    def forward(self, x, adj):
        x = self.lin_in(x)
        for layer in self.lins[:self.num_layers - 1]:
            x = layer(spmm(adj, x))
        x = self.lins[-1](x)
        return x.log_softmax(dim=-1)

class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=300, addbias=True):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
                                          (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        if self.addbias:
            y += self.bias
        y = y.view(outshape)
        return y

def train(args, feat, adj, label, mask, model, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(feat, adj)
    pred, true = out[mask], label[mask]
    loss = F.nll_loss(pred, true)

    # --- Adicionando a regularização L2 para penalizar pesos grandes em relação aos demais ---
    l2_lambda = 0.001  # Define o peso da regularização L2 para controlar a força da regularização. Valores maiores penalizam mais os pesos grandes.
    l2_reg = torch.tensor(0.).to(args.device) # Inicializa tensor de acúmulo da norma L2 dos pesos, que deve estar no mesmo dispositivo (CPU/GPU) que os parâmetros do modelo.
    for param in model.parameters(): # Itera sobre todos os parâmetros (pesos) do modelo.
        l2_reg += torch.norm(param) # Calcula a norma L2 do parâmetro atual (torch.norm(param)) e adiciona ao acumulador l2_reg.
    loss = loss + l2_lambda * l2_reg # Adiciona o termo de regularização L2 à perda original, multiplicado pelo peso l2_lambda.
    # --- Fim da regularização L2 ---

    acc = int((pred.argmax(dim=-1) == true).sum()) / int(mask.sum())
    loss.backward()
    optimizer.step()
    return acc, loss.item()

@torch.no_grad()
def eval(args, feat, adj, model):
    model.eval()
    with torch.no_grad():
        pred = model(feat, adj)
    pred = pred.argmax(dim=-1)
    return pred

class Args:
    path = './data/'
    name = 'Cora'
    logger_path = 'logger/esm'
    dropout = 0.0
    hidden_size = 256
    grid_size = 200
    n_layers = 2
    epochs = 1000
    early_stopping = 100
    seed = 42
    lr = 5e-4

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_disassortative_splits(labels, num_classes, trn_percent=0.6, val_percent=0.2):
    labels, num_classes = labels.cpu(), num_classes.cpu().numpy()
    indices = []
    for i in range(num_classes):
        index = torch.nonzero((labels == i)).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    percls_trn = int(round(trn_percent * (labels.size()[0] / num_classes)))
    val_lb = int(round(val_percent * labels.size()[0]))
    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_mask = index_to_mask(train_index, size=labels.size()[0])
    val_mask = index_to_mask(rest_index[:val_lb], size=labels.size()[0])
    test_mask = index_to_mask(rest_index[val_lb:], size=labels.size()[0])

    return train_mask, val_mask, test_mask
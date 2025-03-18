"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import root_path
import os
from layers.layers import FermiDiracDecoder
import layers.hyp_layers as hyp_layers
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import acc_f1
from utils.cb_loss import CB_loss
from transformers import AutoModel, AutoTokenizer, BertConfig, RobertaTokenizer, RobertaModel

def cb_loss(targets, outputs, beta, gamma, samples_per_class):
    beta = beta
    gamma = gamma
    no_of_classes = 2
    loss_type = "focal"
    return CB_loss(targets, outputs, samples_per_class, no_of_classes, loss_type, beta, gamma)


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        #print('basemodel args feat dim----',args.feat_dim)
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h
    
    def forward(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        #print('basemodel NCModel args feat dim----',args.feat_dim)
        if args.nb_class > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'macro'
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.nb_class)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

        self.beta = args.cb_beta
        self.gamma = args.cb_gamma

        self.stopping_metric = args.stopping_metric


    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)
    
    def forward(self, x, adj, idx):
        embeddings=self.encode(x,adj)
        output = self.decoder.decode(embeddings, adj)
        return output[idx]
    
    # def encode(self, x, adj):
    #     if self.manifold.name == 'Hyperboloid':
    #         o = torch.zeros_like(x)
    #         x = torch.cat([o[:, 0:1], x], dim=1)
    #     h = self.encoder.encode(x, adj)
    #     return h

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        # loss = cb_loss(data['labels'][idx], output, self.beta, self.gamma,\
        #                 data['labels'][idx].unique(return_counts=True)[1].tolist())
        loss=F.nll_loss(output, data['labels'][idx])
        acc, f1, recall = acc_f1(output, data['labels'][idx], split, average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1, 'recall': recall}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1, 'recall': -1, 'loss': np.inf}

    def has_improved(self, m1, m2):
        if self.stopping_metric == "loss":
            return m1["loss"] > m2['loss']
        else:
            return m1[self.stopping_metric] < m2[stopping_metric]


class HyperBertGCN(torch.nn.Module):
    def __init__(self, args):
        super(HyperBertGCN, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.split_gpus = args.split_gpus
        
        self.gcnencoder= getattr(encoders, args.model)(self.c, args)
        self.gcndecoder = model2decoder[args.model](self.c, args)
        self.m = args.m
        self.nb_class = args.nb_class
        self.hidden=args.hidden
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.bert_model = AutoModel.from_pretrained('roberta-base')
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = torch.nn.Linear(self.feat_dim, args.nb_class)
    
    def forward(self, data,idx):#x input_features
        '''if self.split_gpus:
x = x.cuda(0)
x = self.module1(x)
if self.split_gpus:
x = x.cuda(1)
x = self.module2(x)'''
        input_ids, attention_mask = (data['input_ids'][idx]).cuda(1), (data['attention_mask'][idx]).cuda(1)
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            idx=idx.cuda(1)
            print(cls_feats.device,idx.device,(data['cls_feats']).device)
            data['cls_feats']=(data['cls_feats']).cuda(1)
            data['cls_feats'][idx] = cls_feats
        else:
            cls_feats = data['cls_feats'][idx]

        cls_logit = self.classifier(cls_feats)
        cls_pred = torch.nn.Softmax(dim=1)(cls_logit)
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(data['cls_feats'])
            x = torch.cat([o[:, 0:1], data['cls_feats']], dim=1)
            x=x.cuda(0)
        else:
            x=data['cls_feats']

        gcn_hidden = self.gcnencoder.encode(x, (data['adj_train_norm']).cuda(0))
        gcn_logit = self.gcndecoder.decode(gcn_hidden, data['adj_train_norm'].cuda(0))
        idx=idx.cuda(0)
        gcn_pred = torch.nn.Softmax(dim=1)(gcn_logit[idx])
        pred = (gcn_pred.cuda(1) + 1e-10) * self.m + cls_pred * (1 - self.m)
        pred = torch.log(pred)
        return pred




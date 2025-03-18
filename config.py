import os
root_path = os.path.abspath(os.path.dirname(__file__))

import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'seed': (1, 'seed for training'), 
        'gamma': (0.1, 'gamma for lr scheduler'),
        'max_length':(128,'the input length for bert'),
        'lr': (1e-2, 'learning rate'),
        'graph':('all','select different graphs [doc,word,all]'),
        'dataset': ('hcr', 'which dataset to use'),
        'gcn_lr':(1e-3,'gcn rate'),
        'batch_size':(64,'batch_size'),
        'bert_batch':(32,'bert_batch size of updata feature'),
        'bert_lr':(1e-5,'bert rate'),
        'dropout': (0.4, 'dropout probability'),
        'finalmodel':('hbertgcn','can be any of [bert,hbertgcn]'),
        'model': ('HGCN', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'),
        'dim': (256, 'gcn embedding dimension'),
        'manifold': ('PoincareBall', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'),
        'm':(0.7,'the factor balancing BERT and GCN prediction'),
        'pretrained_model': ('bert-base-uncased','roberta-base, roberta-large,bert-base-uncased, bert-large-uncased'),
        'epochs': (200, 'maximum number of epochs to train for'),
        'weight_decay': (5e-4, 'l2 regularization strength'),
        'num_layers': (2, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'optimizer': ('Adam', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'task': ('nc', 'task'),
        'pretrained_bert_ckpt':(None,'pretrained_bert_ckp'),
        'checkpoint_dir':(None,'checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified'),
        'momentum': (0.999, 'momentum in optimizer'),
        'patience': (100, 'patience for early stopping'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'save': (0, '1 to save model and logs and 0 otherwise'),
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'sweep_c': (0, ''),
        'lr_reduce_freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'print_epoch': (True, ''),
        'grad_clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min_epochs': (10, 'do not early stop before min-epochs'),
        'cb_beta': (0.999, 'cb loss beta'),
        'cb_gamma': (2, 'cb loss gamma'),
        'stopping_metric': ('loss', 'stopping criteria'),
        'bert_init':('bert-base-uncased','roberta-base, roberta-large,bert-base-uncased,bert-large-uncased'),
        'hidden':(200,'the hidden embeding dim of bert'),
        'feat_dim':(768,'the embedding output by bert'),
    },
    'model_config': {
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'pretrained-embeddings': (None, 'path to pretrained embeddings (.npy file) for Shallow node classification'),
        'pos_weight': (0, 'whether to upweight positive class in node classification tasks'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n_heads': (8, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'double-precision': ('0', 'whether to use double precision'),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        
        'nb_class':(2,'number of classes to be classified'),
    },
    'data_config': {
        'val_prop': (0.05, 'proportion of validation edges for link prediction'),
        'test_prop': (0.1, 'proportion of test edges for link prediction'),
        'use_feats': (1, 'whether to use node features or not'),
        'normalize_feats': (0, 'whether to normalize input node features'),
        'normalize_adj': (1, 'whether to row-normalize the adjacency matrix'),
        'split_seed': (1234, 'seed for data splits (train/test/val)')
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)


"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
#from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.sparse.linalg import eigsh

def load_data(args, datapath):
    data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed,args.graph)
    data['features'] = process_features(
             data['features'],  args.normalize_feats
    )
    data['adj_all'] = process_graph(
             data['adj_all'],  args.normalize_adj
    )
    data['adj_doc'] = process_graph(
             data['adj_doc'],  args.normalize_adj
    )
    data['adj_word'] = process_graph(
             data['adj_word'],  args.normalize_adj
    )
    return data


# ############### FEATURES PROCESSING ####################################


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def process_features(features,  normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    return  features

def process_graph(adj, normalize_adj):
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### NODE CLASSIFICATION DATA LOADERS ####################################

def load_data_nc(dataset, use_feats, data_path, split_seed,graph):
    if dataset in ['omd','hcr']:
        alladj,docadj,wordadj, features, labels,idx_train, idx_val, idx_test, y_train, y_val, \
            y_test, train_mask, val_mask, test_mask, train_size, test_size= load_corpus(dataset,graph)
        labels = torch.LongTensor(labels)
        data = {'adj_all': alladj,'adj_doc': docadj,'adj_word': wordadj, 'features': features, 'labels': labels,\
             'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test,\
                'y_train':y_train,'y_test':y_test,'y_val':y_val,\
                    'train_mask':train_mask,'val_mask':val_mask,'test_mask':test_mask,\
                        'train_size':train_size,'test_size':test_size}
    return data


# ############### DATASETS ####################################

def load_corpus(dataset_str,graph):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally','docx','docy']
    objects = []
    for i in range(len(names)):
        with open("data/{}/ind.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    names1=['all','doc','word']
    for i in range(len(names1)):
        with open("data/{}/ind.{}.adj".format(dataset_str, names1[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally,docx,docy,allgraph,docgraph,wordgraph = tuple(objects)
    if graph=='all' or graph=='word':
        features = sp.vstack((allx, tx)).tolil()
        labels = np.vstack((ally, ty))
    else:
        features = sp.vstack((docx, tx)).tolil()
        labels = np.vstack((docy, ty))
    train_idx_orig = parse_index_file(
        "data/{}/train.index".format(dataset_str))
    train_size = len(train_idx_orig)
    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    if graph=='all' or graph=='word':
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + val_size)
        idx_test = range(allx.shape[0], allx.shape[0] + test_size)
    else:
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + val_size)
        idx_test = range(len(y) + val_size, len(y) + val_size + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    alladj = allgraph + allgraph.T.multiply(allgraph.T > allgraph) - allgraph.multiply(allgraph.T > allgraph)
    docadj = docgraph + docgraph.T.multiply(docgraph.T > docgraph) - docgraph.multiply(docgraph.T > docgraph)
    wordadj = wordgraph + wordgraph.T.multiply(wordgraph.T > wordgraph) - wordgraph.multiply(wordgraph.T > wordgraph)
    labels1 = np.argmax(labels, 1)
    return alladj,docadj,wordadj, features, labels1,idx_train, idx_val, idx_test,y_train, y_val, y_test, \
        train_mask, val_mask, test_mask, train_size, test_size

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)
def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index








from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time
import torch.nn.functional as F
import numpy as np
import optimizers
import torch
from config import parser
from models.base_models import HyperBertGCN
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics
import torch.utils.data as Data
import shutil
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss,Precision,Recall,Fbeta
import sys
from sklearn.metrics import accuracy_score,recall_score,f1_score

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3600"

args = parser.parse_args()
max_length = args.max_length
batch_size = args.batch_size
m = args.m
nb_epochs = args.epochs
bert_init = args.bert_init
pretrained_bert_ckpt = args.pretrained_bert_ckpt
dataset = args.dataset
checkpoint_dir = args.checkpoint_dir
gcn_layers = args.num_layers
gcn_model=args.model
n_hidden = args.dim
heads = args.n_heads
dropout = args.dropout
gcn_lr = args.gcn_lr
bert_lr = args.bert_lr

if checkpoint_dir is None:
    ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, gcn_model, dataset)
else:
    ckpt_dir = checkpoint_dir
os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if int(args.double_precision):
    torch.set_default_dtype(torch.float64)
if int(args.cuda) >= 0:
    torch.cuda.manual_seed(args.seed)
if not args.lr_reduce_freq:
    args.lr_reduce_freq = args.epochs


logger = logging.getLogger('training logger')
logger.setLevel(logging.INFO)

cpu = torch.device('cpu')
args.device=gpu = torch.device('cuda:0')

logger.info('arguments:')
logger.info(str(args))
logger.info('checkpoints will be saved in {}'.format(ckpt_dir))

#Data Preprocess
data = load_data(args, os.path.join("data/", args.dataset))
#print('data----',data)
nb_node = data['adj_all'].shape[0]
nb_train, nb_val, nb_test = data['train_mask'].sum(), data['val_mask'].sum(), data['test_mask'].sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = data['y_train'].shape[1]

args.n_nodes, args.feat_dim1 = data['features'].shape
if args.finalmodel=='bert':
    Model = BertClassifier
    args.n_classes = int(data['labels'].max() + 1)
    logging.info(f'Num classes: {args.n_classes}')
else args.finalmodel=='hbertgcn':
    Model=HyperBertGCN
    args.n_classes = int(data['labels'].max() + 1)
    logging.info(f'Num classes: {args.n_classes}')



model = Model(args)
logging.info(str(model))

if pretrained_bert_ckpt is not None:
    ckpt = torch.load(pretrained_bert_ckpt, map_location=gpu)
    model.bert_model.load_state_dict(ckpt['bert_model'])
    model.classifier.load_state_dict(ckpt['classifier'])

# load documents and compute input encodings
corpse_file = './data/' + dataset +'/corpus/shuffle.txt'
with open(corpse_file, 'r') as f:
    text = f.read()
    #text = text.replace('\\', '')
    text = text.split(os.linesep)


def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
#     print(input.keys())
    return input.input_ids, input.attention_mask


input_ids, attention_mask = encode_input(text, model.tokenizer)
#print('input_ids----',input_ids.shape)
if args.graph=='all' or args.graph=='word':
    input_ids = torch.cat([input_ids[:-nb_test], torch.zeros((nb_word, max_length), dtype=torch.long), input_ids[-nb_test:]])
    attention_mask = torch.cat([attention_mask[:-nb_test], torch.zeros((nb_word, max_length), dtype=torch.long), attention_mask[-nb_test:]])


# transform one-hot label to class ID for pytorch computation
y = data['y_train'] + data['y_test'] + data['y_val']
y_train = data['y_train'].argmax(axis=1)
y = y.argmax(axis=1)

doc_mask  = data['train_mask'] + data['val_mask'] + data['test_mask']

data['input_ids']  = input_ids
data['attention_mask']=attention_mask
if args.graph=='all' or args.graph=='word':
    data['cls_feats'] = torch.zeros((nb_node, model.feat_dim))
else:
    data['cls_feats'] = torch.zeros((data['adj_doc'].shape[0], model.feat_dim))


# create index loader
if args.graph=='word' or args.graph=='all':
    train_idx = Data.TensorDataset(torch.arange(0, nb_train, dtype=torch.long))
    val_idx = Data.TensorDataset(torch.arange(nb_train, nb_train + nb_val, dtype=torch.long))
    test_idx = Data.TensorDataset(torch.arange(nb_node-nb_test, nb_node, dtype=torch.long))
    doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

    idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
    idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
    idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
    idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)
else:
    train_idx = Data.TensorDataset(torch.arange(0, nb_train, dtype=torch.long))
    val_idx = Data.TensorDataset(torch.arange(nb_train, nb_train + nb_val, dtype=torch.long))
    test_idx = Data.TensorDataset(torch.arange(nb_train + nb_val, nb_train + nb_val + nb_test, dtype=torch.long))
    doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

    idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
    idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
    idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
    idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)

def update_feature():
    global model, data, doc_mask
    # no gradient needed, uses a large batchsize to speed up the process
    dataloader = Data.DataLoader(
        Data.TensorDataset(data['input_ids'][doc_mask], data['attention_mask'][doc_mask]),
        batch_size=args.bert_batch
    )
    with torch.no_grad():
        model = model.to(gpu)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):
            input_ids, attention_mask = [x.cuda(0) for x in batch]
            output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            cls_list.append(output)
        cls_feat = torch.cat(cls_list, axis=0)
    for x, val in data.items():
        if torch.is_tensor(data[x]):
            data[x] = data[x].to(args.device)
    data['cls_feats'][doc_mask] = cls_feat
    return data


if args.finalmodel=='bert':
    optimizer = torch.optim.Adam(model.parameters(), lr=bert_lr)
elif args.finalmodel=='hbertgcn':
    optimizer = torch.optim.Adam([
        {'params': model.bert_model.parameters(), 'lr': bert_lr},
        {'params': model.classifier.parameters(), 'lr': bert_lr},
        {'params': model.gcnencoder.parameters(), 'lr': gcn_lr},
    ], lr=args.lr,weight_decay=args.weight_decay
)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=bert_lr)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=args.gamma)

def train_step(engine, batch):
    global model, data, optimizer
    model.train()
    model = model.to(gpu)
    optimizer.zero_grad()
    (idx, ) = [x for x in batch]
    optimizer.zero_grad()
    train_mask=torch.tensor(data['train_mask'][idx],dtype=torch.bool)
    for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
    y_pred=model.forward(data, idx)[train_mask]
    y_true = ((torch.tensor((data['y_train']).argmax(axis=1)))[idx][train_mask])
    loss = F.nll_loss(y_pred, (y_true).to(args.device))
    loss.backward()
    optimizer.step()
    data['cls_feats'].detach_()
    with torch.no_grad():
        if train_mask.sum() > 0:
            y_true = y_true.detach().cpu()
            y_pred = y_pred.argmax(axis=1).detach().cpu()
            train_acc = accuracy_score(y_true, y_pred)
            train_f1=f1_score(y_true, y_pred)
            train_recall=recall_score(y_true, y_pred)
        else:
            train_acc = 1
            train_f1=1
            train_recall=1
    return loss, train_acc, train_f1, train_recall

trainer = Engine(train_step)


@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    scheduler.step()
    update_feature()
    torch.cuda.empty_cache()

def test_step(engine, batch):
    global model, data
    with torch.no_grad():
        model.eval()
        model = model.to(gpu)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
        (idx, ) = [x.to(gpu) for x in batch]
        y_pred = model(data, idx)
        y_true = data['labels'][idx]
        return y_pred, y_true
    

P = Precision(average=False)
R = Recall(average=False)
F1 = Fbeta(beta=1.0, average='macro', precision=P, recall=R)
evaluator = Engine(test_step)
metrics={
    'acc': Accuracy(),
    'nll': Loss(torch.nn.NLLLoss()),
    'recall': Recall(),
    'f1': F1
}

for n, f in metrics.items():
    f.attach(evaluator, n)

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(idx_loader_train)
    metrics = evaluator.state.metrics
    train_nll,train_acc, train_f1 = metrics["nll"],metrics["acc"], metrics['f1']
    evaluator.run(idx_loader_val)
    metrics = evaluator.state.metrics
    val_nll,val_acc, val_f1 = metrics["nll"],metrics["acc"], metrics['f1']
    evaluator.run(idx_loader_test)
    metrics = evaluator.state.metrics
    test_nll,test_acc, test_f1 = metrics["nll"],metrics["acc"], metrics['f1']
    print(
        "Epoch: {} Train loss: {:.4f} Train acc: {:.4f} Train F1:{:.4f}"
        .format(trainer.state.epoch, train_nll,train_acc,train_f1)
    )
    print("val oss: {:.4f} Val acc: {:.4f} Val F1:{:.4f} ".format( val_nll,  val_acc, val_f1))
    print("test loss: {:.4f}  Test acc: {:.4f} Test F1:{:.4f}".format(test_nll, test_acc, test_f1))
    if val_acc > log_training_results.best_val_acc:
        logger.info("New checkpoint")
        if args.finalmodel=='hbertgcn':
            torch.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'gcnencoder': model.gcnencoder.state_dict(),
                'gcndncoder': model.gcndecoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        elif args.finalmodel=='bert':
            torch.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        else:
            torch.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_acc = val_acc


log_training_results.best_val_acc = 0
data = update_feature()
torch.cuda.empty_cache()
trainer.run(idx_loader, max_epochs=nb_epochs)

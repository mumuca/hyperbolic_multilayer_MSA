
import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import pandas as pd
#from utils.utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine
from config import root_path
import re
# if len(sys.argv) != 2:
# 	sys.exit("Use: python build_graph.py <dataset>")
np.random.seed(1)

datasets = ['hcr', 'omd']
# build corpus
#dataset = sys.argv[1]
dataset='hcr'

if dataset not in datasets:
	sys.exit("wrong dataset name")

word_embeddings_dim = 300
word_vector_map = {}

# shulffing
doc_name_list = []
doc_train_list = []
doc_test_list = []

train_ids=[]
for line in open('data/' + dataset + '/train.index','r'):
    line=line.splitlines()
    line=''.join(line)
    line=int(line)
    train_ids.append(line)
print(train_ids)


test_ids=[] 
for line in open('data/' + dataset + '/test.index','r'):
    line=line.splitlines()
    line=''.join(line)
    line=int(line)
    test_ids.append(line)
print(test_ids)

ids=train_ids+test_ids
neg = pd.read_excel(os.path.join(root_path, 'data/'+dataset+'/'+dataset+'2-neg-con.xls'), header=None)
pos = pd.read_excel(os.path.join(root_path, 'data/'+dataset+'/'+dataset+'2-pos-con.xls'), header=None)
combined = np.concatenate((neg[0], pos[0])) 
doc_content_list=combined.tolist()
doc_name_list=list(range(0,len(neg[0])+len(pos[0])))
doc_name_list=[str(x) for x in doc_name_list]

shuffle_doc_name_list = []
shuffle_doc_words_list = []
for id in ids:
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)
f = open('data/' + dataset + '/shuffle.txt', 'w')
f.write(shuffle_doc_name_str)
f.close()

# build vocab
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
regex_str = [
    emoticons_str,
    r'[w]/',
    r'<[^>]+>',  # HTML tags
    #r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    #r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    #r'{^\w|^\s}{1,}',
    #r'[.]+',
    r'[!.?]+',
    r'[\\]+',
    r'[//]+',
    r'(?:\S)',  # anything else
]

regex_str_remove=[
     r'[R][T]',
     r'(?:@[\w_]+)',  # @-mentions
     r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
    ]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

textnew = []
for line in shuffle_doc_words_list:
    line=re.sub(r'(' + '|'.join(regex_str_remove) + ')',"",line)
    linenew=preprocess(line)
    textnew.append(linenew)
print('len(textnew)-------',len(textnew))

f=open('data/' + dataset + '/corpus/shuffle.txt', 'w')
for line in textnew:
    # f.writelines(line)
    # f.write(os.linesep)
    for term in line:
        f.write(term+' ')
    f.write(os.linesep)
f.close()


word_freq = {}
word_set = set()
for doc_words in textnew:
    for word in doc_words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)

word_doc_list = {}

for i in range(len(textnew)):
    doc_words = textnew[i]
    appeared = set()
    for word in doc_words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)

f = open('data/' + dataset + '/corpus/vocab.txt', 'w')
f.write(vocab_str)
f.close()

train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size 

real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)

f = open('data/' + dataset + '/real_train.name', 'w')
f.write(real_train_doc_names_str)
f.close()

row_x = []
col_x = []
data_x = []
for i in range(real_train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = textnew[i]
    doc_len = len(doc_words)
    for word in doc_words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        data_x.append(doc_vec[j] / doc_len)

x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, word_embeddings_dim))

label_list = list(range(2))
y = []
for i in range(real_train_size):
    doc_meta = shuffle_doc_name_list[i]
    if int(doc_meta)<len(neg[0]):
        label=0
    else:
        label=1
    one_hot = [0 for l in range(2)]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)

# tx: feature vectors of test docs, no initial features
test_size = len(test_ids)

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = textnew[i + train_size]
    doc_len = len(doc_words)
    for word in doc_words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        data_tx.append(doc_vec[j] / doc_len)

tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

ty = []
for i in range(test_size):
    doc_meta = shuffle_doc_name_list[i + train_size]
    if int(doc_meta)<len(neg[0]):
        label=0
    else:
        label=1
    one_hot = [0 for l in range(2)]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)

# allx: the the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> words

word_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, word_embeddings_dim))

for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []
data_docx=[]
for i in range(train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = textnew[i]
    #words = doc_words.split()
    doc_len = len(doc_words)
    for word in doc_words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)
    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
        data_docx.append(doc_vec[j] / doc_len)

data_docx = np.array(data_docx)
docx=sp.csr_matrix(
    (data_docx, (np.array(row_allx), np.array(col_allx))), shape=(train_size, word_embeddings_dim))


for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))
row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)
allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

ally = []
docy=[]
for i in range(train_size):
    doc_meta = shuffle_doc_name_list[i]
    if int(doc_meta)<len(neg[0]):
        label=0
    else:
        label=1
    one_hot = [0 for l in range(2)]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)
    docy.append(one_hot)
docy = np.array(docy)

for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)

print('shape of all objects-----',x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape, docx.shape, docy.shape)

'''
Doc word heterogeneous graph
'''

# word co-occurence with context windows
window_size = 20
windows = []

for doc_words in textnew:
    #words = doc_words.split()
    length = len(doc_words)
    if length <= window_size:
        windows.append(doc_words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = doc_words[j: j + window_size]
            windows.append(window)
            # print(window)

word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            #print('word_j',word_j)
            #print(word_id_map[word_j])
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

row = []
col = []
weight = []
row1 = []
col1 = []
weight1 = []


# pmi as weights

num_window = len(windows)

for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)

# doc word frequency
doc_word_freq = {}

for doc_id in range(len(shuffle_doc_words_list)):
    doc_words = textnew[doc_id]
    #words = doc_words.split()
    for word in doc_words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

for i in range(len(shuffle_doc_words_list)):
    doc_words = textnew[i]
    #words = doc_words.split()
    doc_word_set = set()
    for word in doc_words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        doc_word_set.add(word)

node_size = train_size + vocab_size + test_size
word_doc_adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))


#build social graph
mmuu=np.loadtxt('data/' + dataset + '/'+dataset+'mmuu.txt') # load social graph file
for i in range(len(shuffle_doc_name_list)):
    for j in range(len(shuffle_doc_name_list)):
        i_new=int(shuffle_doc_name_list[i])
        j_new=int(shuffle_doc_name_list[j])
        if mmuu[i_new,j_new]!=0:
            if i<train_size:
                row.append(i)
                #col.append[j]
            else:
                row.append(i+vocab_size)
            if j<train_size:
                col.append(j)
            else:
                col.append(j+vocab_size)
            weight.append(mmuu[i_new,j_new])

node_size = train_size + vocab_size + test_size
alladj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

for i in range(len(shuffle_doc_name_list)):
    for j in range(len(shuffle_doc_name_list)):
        i_new=int(shuffle_doc_name_list[i])
        j_new=int(shuffle_doc_name_list[j])
        if mmuu[i_new,j_new]!=0:   
            row1.append(i)
            col1.append(j)
            weight1.append(mmuu[i_new,j_new])

node_size1 = train_size + test_size
doc_adj = sp.csr_matrix(
    (weight1, (row1, col1)), shape=(node_size1, node_size1))

# dump objects
f = open("data/{}/ind.x".format(dataset), 'wb')
pkl.dump(x, f)
f.close()

f = open("data/{}/ind.y".format(dataset), 'wb')
pkl.dump(y, f)
f.close()

f = open("data/{}/ind.tx".format(dataset), 'wb')
pkl.dump(tx, f)
f.close()

f = open("data/{}/ind.ty".format(dataset), 'wb')
pkl.dump(ty, f)
f.close()

f = open("data/{}/ind.allx".format(dataset), 'wb')
pkl.dump(allx, f)
f.close()

f = open("data/{}/ind.ally".format(dataset), 'wb')
pkl.dump(ally, f)
f.close()

f = open("data/{}/ind.docx".format(dataset), 'wb')
pkl.dump(docx, f)
f.close()

f = open("data/{}/ind.docy".format(dataset), 'wb')
pkl.dump(docy, f)
f.close()

f = open("data/{}/ind.word.adj".format(dataset), 'wb')
pkl.dump(word_doc_adj, f)
f.close()

f = open("data/{}/ind.doc.adj".format(dataset), 'wb')
pkl.dump(doc_adj, f)
f.close()

f = open("data/{}/ind.all.adj".format(dataset), 'wb')
pkl.dump(alladj, f)
f.close()
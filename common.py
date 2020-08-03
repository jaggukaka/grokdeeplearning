import numpy as np
from os import walk
from collections import Counter
import math, warnings

np.seterr(all='raise')

POS_FILE_PATH = './aclImdb/train/pos/'
NEG_FILE_PATH = './aclImdb/train/neg/'

TEST_POS = './aclImdb/test/pos/'
TEST_NEG = './aclImdb/test/neg/'

def get_word_index(path):
    word2index = {}
    with open(path, '+r') as f:
        i = 0
        while 1:
            line = f.readline()
            if not line:
                break
            word2index[line.strip()] = i
            i += 1
    return word2index


def get_word_index_imdb():
    return get_word_index('./aclImdb/imdb.vocab')


def get_files(path):
    f = []
    for (_,_,filenames) in walk(path):
        f.extend(filenames)
    return f

def get_content(path):
    with open(path, '+r') as f:
        line = f.readline()
        if line:
            return line
    return ""

def sigmoid(x):
    x[x < -100] = -100
    x[x > 100] = 100
    try:
        return 1/(1 + np.exp(-x))
    except Exception as e:
        print(x, e)

def sigmoid_deriv(x):
    return x*(1-x)

def similar(target, w1, w2, word2index):
    target_index = word2index[target]
    scores = Counter()
    target_w = w1[target_index]
    for word,index in word2index.items():
        scores[word] = -math.sqrt(sum((w1[index] - target_w)**2))

    return scores.most_common(10)

def normalize(w1):
    norms = np.sum(w1**2, axis=1)
    norms.resize(norms.shape[0], 1)

    normw = w1*norms

    return normw

def analogy(pos, neg, w1, w2, word2index):
    normw = normalize(w1)
    q_vector = np.zeros((w1[0].shape))
    for word in pos:
        q_vector += normw[word2index[word]]
    for word in neg:
        q_vector -= normw[word2index[word]]

    scores = Counter()
    for word, index in word2index.items():
        scores[word] = -math.sqrt(sum((w1[index] - q_vector)**2))
    
    return scores.most_common(10)
    
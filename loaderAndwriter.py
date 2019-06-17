import codecs
import numpy as np
import os

def load_word_embedding(embedding_path, word_idx):
    with codecs.open(embedding_path, 'r', 'utf-8') as f:
        vecs = []
        for line in f:
            line = line.strip()
            if len(line.split(" ")) == 2:
                continue
            info = line.split(' ')
            word = info[0]
            vec = [float(v) for v in info[1:]]
            if len(vec) != 300:
                continue
            vecs.append(vec)
            word_idx[word] = len(word_idx.keys()) + 1  #  + 1 is due to that we already have an unknown word

    return np.array(vecs)

def write_word_idx(word_idx, path):
    dir = path[:path.rfind('/')]
    if not os.path.exists(dir):
        os.makedirs(dir)

    with codecs.open(path, 'w', 'utf-8') as f:
        for word in word_idx:
            f.write(str(word)+" "+str(word_idx[word])+'\n')

def read_word_idx_from_file(path, if_key_is_int=False):
    word_idx = {}
    with codecs.open(path, 'r', 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            info = line.strip().split(" ")
            if len(info) != 2:
                word_idx[' '] = int(info[0])
            else:
                if if_key_is_int:
                    word_idx[int(info[0])] = int(info[1])
                else:
                    word_idx[info[0]] = int(info[1])
    return word_idx


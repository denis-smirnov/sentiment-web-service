import numpy as np


def load_embeddings(filename):
    with open(filename) as f:
        embedding_dict_size, embedding_dim = (int(d) for d in f.readline().split())
        embeddings = dict()
        for next_embed in f:
            line_values = next_embed.split()
            embeddings[line_values[0]] = [np.float(v) for v in line_values[1:]]
        return embedding_dict_size, embedding_dim, embeddings

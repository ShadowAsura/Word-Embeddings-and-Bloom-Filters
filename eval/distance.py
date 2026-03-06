# From https://github.com/stanfordnlp/GloVe
# Adapted to work with JSON iterative vectors

import argparse
import numpy as np
import json
import os

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', default=6, type=int, help='Window size for embeddings')
    parser.add_argument('--iteration', default=399, type=int, help='Iteration number to load')
    parser.add_argument('--vector_dir', default='data/iterative_vectors', type=str, help='Directory containing vector files')
    args = parser.parse_args()

    # Load vectors from JSON
    vector_file = os.path.join(args.vector_dir, f'window_{args.window}_iter_{args.iteration}.json')
    
    if not os.path.exists(vector_file):
        raise FileNotFoundError(f"Vector file not found: {vector_file}")
    
    with open(vector_file, 'r') as f:
        vectors = json.load(f)
    
    # Build vocabulary from JSON keys
    words = sorted(list(vectors.keys()))
    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    # Build embedding matrix from JSON values
    vector_dim = len(vectors[words[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word in vocab:
            W[vocab[word], :] = v

    # normalize each word vector to unit length
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    d[d == 0] = 1  # avoid division by zero
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab)


def distance(W, vocab, ivocab, input_term):
    for idx, term in enumerate(input_term.split(' ')):
        if term in vocab:
            print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result = np.copy(W[vocab[term], :])
            else:
                vec_result += W[vocab[term], :]
        else:
            print('Word: %s  Out of dictionary!\n' % term)
            return

    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2,) ** (0.5))
    vec_norm = (vec_result.T / d).T

    dist = np.dot(W, vec_norm.T)

    for term in input_term.split(' '):
        index = vocab[term]
        dist[index] = -np.inf

    a = np.argsort(-dist)[:N]

    print("\n                               Word       Cosine distance\n")
    print("---------------------------------------------------------\n")
    for x in a:
        print("%35s\t\t%f\n" % (ivocab[x], dist[x]))


if __name__ == "__main__":
    N = 100 # number of closest words that will be shown
    W, vocab, ivocab = generate()
    while True:
        input_term = input("\nEnter word or sentence (EXIT to break): ")
        if input_term == 'EXIT':
            break
        else:
            distance(W, vocab, ivocab, input_term)

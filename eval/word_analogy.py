# From https://github.com/stanfordnlp/GloVe
# Adapted to work with JSON iterative vectors

import argparse
import numpy as np
import json
import os

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', default=6, type=int, help='Window size for embeddings')
    iteration = 2 # We load the latest iteration generated
    parser.add_argument('--iteration', default=iteration, type=int, help='Iteration number to load')
    parser.add_argument('--vector_dir', default=None, type=str, help='Directory containing vector files')
    args = parser.parse_args()

    # Auto-detect vector directory if not provided
    if args.vector_dir is None:
        # Check if we're in eval/ directory
        if os.path.exists('data/iterative_vectors'):
            args.vector_dir = 'data/iterative_vectors'
        elif os.path.exists('../data/iterative_vectors'):
            args.vector_dir = '../data/iterative_vectors'
        else:
            raise FileNotFoundError("Could not find data/iterative_vectors directory. Try running from workspace root or eval/")
    
    # Load vectors from JSON
    vector_file = os.path.join(args.vector_dir, f'window_{args.window}_iter_{args.iteration}_v3_200bit.json')
    
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

    # normalize each word vector to unit length for stable cosine distances
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    d[d == 0] = 1  # avoid division by zero
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab)


def distance(W, vocab, ivocab, input_term):
    """
    input_term: string of 3 words 'king man queen'
    """
    # Split input into words and convert to indices
    words = input_term.split()
    if len(words) < 3:
        print(f"Only {len(words)} words were entered.. three words are needed at the input to perform the calculation\n")
        return
    
    indices = [vocab.get(word) for word in words[:3]]
    if None in indices:
        missing = [words[i] for i, idx in enumerate(indices) if idx is None]
        print(f"Words not in vocabulary: {missing}\n")
        return
    
    # Compute analogy vector: word2 - word1 + word3
    vec = W[indices[1]] - W[indices[0]] + W[indices[2]]
    
    # Normalize the analogy vector to unit length
    d = (np.sum(vec ** 2) ** (0.5))
    vec_norm = (vec / d) if d > 0 else vec
    
    # Compute cosine distances to all words
    dist = np.dot(W, vec_norm)
    
    # Set input words to -inf so they don't rank as answers
    for idx in indices:
        dist[idx] = -np.inf
    
    # Get top 100 closest words
    most_similar = np.argsort(-dist)[:100]
    
    print(f"\nAnalogy: {words[0]} is to {words[1]} as {words[2]} is to ___?")
    print("Top 10 closest words:")
    for rank, idx in enumerate(most_similar[:10], 1):
        print(f"{rank:2}. {ivocab[idx]:20} (similarity={dist[idx]:.4f})")


if __name__ == "__main__":
    N = 100;          # number of closest words that will be shown
    W, vocab, ivocab = generate()
    distance(W, vocab, ivocab, "king man queen")


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
import os
import argparse

def plot_vectors_pca(vector_file_path, words_to_plot, output_filename="pca_plot.png"):
    """
    Loads word vectors, performs PCA, and plots the specified words.

    Args:
        vector_file_path (str): Path to the JSON file containing word vectors.
        words_to_plot (list): A list of words to include in the PCA plot.
        output_filename (str): Name of the file to save the plot.
    """
    if not os.path.exists(vector_file_path):
        print(f"Error: Vector file not found at {vector_file_path}")
        return

    with open(vector_file_path, 'r') as f:
        all_vectors = json.load(f)

    # Filter for words we want to plot
    filtered_words = [word for word in words_to_plot if word in all_vectors]
    if not filtered_words:
        print(f"None of the specified words ({words_to_plot}) found in {vector_file_path}.")
        return

    vectors = np.array([all_vectors[word] for word in filtered_words])
    labels = filtered_words

    # Perform PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    plt.scatter(components[:, 0], components[:, 1], alpha=0.7)

    # Annotate points with word labels
    for i, word in enumerate(labels):
        plt.annotate(word, (components[i, 0], components[i, 1]), textcoords="offset points", xytext=(5, 5), ha='center')

    plt.title(f"PCA of Word Embeddings from {os.path.basename(vector_file_path)}")
    plt.xlabel(f"Principal Component 1 (explained variance: {pca.explained_variance_ratio_[0]:.2f})")
    plt.ylabel(f"Principal Component 2 (explained variance: {pca.explained_variance_ratio_[1]:.2f})")
    plt.grid(True)

    # Ensure output directory exists
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    full_output_path = os.path.join(output_dir, output_filename)

    plt.savefig(full_output_path, dpi=300)
    plt.show()
    print(f"PCA plot saved to {full_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PCA plots for word embeddings.")
    parser.add_argument('--iteration', type=int, default=2, help="Iteration number of the vector file to load.")
    parser.add_argument('--bits', type=int, default=200, help="Bit size of the vectors.")
    parser.add_argument('--window', type=int, default=6, help="Window size used for embeddings.")
    parser.add_argument('--words', nargs='+', default=["king", "queen", "man", "woman", "prince", "princess", "boy", "girl"],
                        help="List of words to plot. Default: king queen man woman prince princess boy girl")
    
    args = parser.parse_args()

    vector_file_name = f'window_{args.window}_iter_{args.iteration}_v3_{args.bits}bit.json'
    vector_file_path = os.path.join('data/iterative_vectors', vector_file_name)

    plot_vectors_pca(vector_file_path, args.words, output_filename=f"pca_plot_iter_{args.iteration}.png")
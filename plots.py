# GENERATES All SPECIFIED PLOTS
# CREATES EMBDEDDING FILE FOR WORDS IF NOT MADE YET

# Stores all output in /results/plots
# Sset true and false to run specific plots
# Need to implemnt one more plot

SOURCE_DATA_PATH = 'data/iterative_vectors/' 
PROCESSED_DATA_DIR = 'data/' 

WANTED_WORDS = ['brother', 'daughter', 'sister', 'son']
WINDOW_SIZE = 4
ITERATION_TO_PLOT = 200 

RUN_CONFIG = {
    "PCA_2D": True,         
    "PCA_3D_POINTS": True,  
    "PCA_3D_VECTORS": True, 
    "UMAP_2D": False,        
    "UMAP_3D": False,        
    "STABILITY": True       
}
# ==========================================

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap
import random
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

os.chdir(Path(__file__).parent.resolve())

def ensure_consolidated_data(words, window, iterative_dir):
    prefix = ''.join([w[0].lower() for w in words])
    output_file = os.path.join(PROCESSED_DATA_DIR, f'{prefix}_iteration_window_{window}.json')
    
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                json.load(f)
            print(f"Using existing valid file: {output_file}")
            return output_file
        except:
            print(f"Detected corrupted consolidated file. Rebuilding...")
            os.remove(output_file)

    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    print(f"Scanning {iterative_dir} for window_{window}...")
    embeddings = {word: [] for word in words}
    pattern = re.compile(rf'^window_{window}_iter_(\d+)\.json$')

    file_names = sorted([f for f in os.listdir(iterative_dir) if pattern.match(f)], 
                        key=lambda x: int(pattern.match(x).group(1)))

    for f_name in file_names:
        file_path = os.path.join(iterative_dir, f_name)
        try:
            with open(file_path, 'r') as f:
                vectors = json.load(f)
                for word in words:
                    if word in vectors:
                        embeddings[word].append(vectors[word])
        except:
            continue 

    with open(output_file, 'w+') as f:
        json.dump(embeddings, f)
    
    print(f"Successfully created consolidated file: {output_file}")
    return output_file

def get_output_params(words):
    prefix = ''.join([w[0].lower() for w in words])
    path = f"./results/plots/{prefix}/"
    os.makedirs(path, exist_ok=True)
    return path, prefix

def plot_pca_2d(data, words, window):
    out_dir, prefix = get_output_params(words)
    points = []
    for word in words:
        points.extend([word, np.array(row)] for row in data[word])
    random.shuffle(points)
    labels = np.array([p[0] for p in points])
    lookup, clabels = np.unique(labels, return_inverse=True)
    vectors = np.array([p[1] for p in points])
    
    pca = PCA(n_components=2)
    feat = pca.fit_transform(vectors)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(feat[:,0], feat[:,1], c=clabels, cmap='tab10', s=5, marker='.')
    
    ax.set_xlabel('PC 1 (%.2f%%)' % (pca.explained_variance_ratio_[0]*100), fontsize=8)
    ax.set_ylabel('PC 2 (%.2f%%)' % (pca.explained_variance_ratio_[1]*100), fontsize=8)
    
    word_list_str = ', '.join([w.capitalize() for w in words])
    ax.set_title(f"PCA of Words: {word_list_str}\nWindow = {window}", fontsize=10)
    ax.legend([mpatches.Patch(color=plt.cm.tab10(i)) for i in range(len(words))], lookup, fontsize=8)
    
    plt.tight_layout()
    save_path = f"{out_dir}{prefix}_iteration_snapshots_window_{window}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Finished 2D PCA plot: {save_path}")

def plot_pca_3d_points(data, words, window):
    out_dir, prefix = get_output_params(words)
    points = []
    for word in words:
        points.extend([word, np.array(row)] for row in data[word])
    labels = np.array([p[0] for p in points])
    lookup, clabels = np.unique(labels, return_inverse=True)
    vectors = np.array([p[1] for p in points])
    
    pca = PCA(n_components=3)
    feat = pca.fit_transform(vectors)
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feat[:,0], feat[:,1], feat[:,2], c=clabels, cmap='tab10', s=5)
    
    ax.set_title(f"3D PCA Points: {', '.join(words)}\nWindow = {window}", fontsize=10)
    ax.legend([mpatches.Patch(color=plt.cm.tab10(i)) for i in range(len(words))], lookup, fontsize=8)
    
    plt.tight_layout()
    save_path = f"{out_dir}{prefix}_3D_Points_window_{window}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Finished 3D PCA Points plot: {save_path}")

def plot_pca_3d_vectors(data_dir, words, window, iteration):
    out_dir, prefix = get_output_params(words)
    file_name = f'window_{window}_iter_{iteration}.json'
    path = os.path.join(data_dir, file_name)
    if not os.path.exists(path): return
    try:
        with open(path, 'r') as f: all_vectors = json.load(f)
    except: return
    
    filtered = [np.array(all_vectors[w]) for w in words if w in all_vectors]
    if not filtered: return
    
    pca = PCA(n_components=3)
    vectors = pca.fit_transform(np.array(filtered))
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.pane.set_facecolor((1, 1, 1, 1))
    
    colors = ["red", "blue", "green", "orange", "purple", "brown"]
    vec_mag = 0
    for i, vec in enumerate(vectors):
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=colors[i % 6], length=1.0, arrow_length_ratio=0.1, label=words[i])
        ax.text(vec[0] * 1.1, vec[1] * 1.1, vec[2] * 1.1, words[i], color='black', fontsize=8)
        vec_mag = max(vec_mag, np.linalg.norm(vec))
    
    limit = vec_mag + 0.25
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(-limit, limit)
    ax.set_title(f"3D PCA Vectors: {', '.join(words)}\nIteration {iteration}", fontsize=10)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    save_path = f"{out_dir}{prefix}_3D_Plot_window_{window}_iteration_{iteration}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Finished 3D Vector plot: {save_path}")

def plot_umap_2d(data, words, window):
    out_dir, _ = get_output_params(words)
    for word in words:
        vectors = np.array(data[word])
        embedding = umap.UMAP(n_components=2).fit_transform(vectors)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(embedding[:, 0], embedding[:, 1], s=5, label=word)
        ax.set_title(f"2D UMAP: '{word}'\nWindow = {window}", fontsize=10)
        ax.legend(fontsize=8)
        
        plt.tight_layout()
        save_path = f"{out_dir}{word}_umap_2d_window_{window}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Finished 2D UMAP plot for '{word}': {save_path}")

def plot_umap_3d(data, words, window):
    out_dir, _ = get_output_params(words)
    for word in words:
        vectors = np.array(data[word])
        embedding = umap.UMAP(n_components=3).fit_transform(vectors)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=5, label=word)
        ax.set_title(f"3D UMAP: '{word}'\nWindow = {window}", fontsize=10)
        ax.legend(fontsize=8)
        
        plt.tight_layout()
        save_path = f"{out_dir}{word}_umap_3d_window_{window}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Finished 3D UMAP plot for '{word}': {save_path}")

def plot_stability(data_dir, words, window):
    out_dir, _ = get_output_params(words)
    pattern = re.compile(rf'^window_{window}_iter_(\d+)\.json$')
    files = sorted([f for f in os.listdir(data_dir) if pattern.match(f)], key=lambda x: int(pattern.match(x).group(1)))
    
    for word in words:
        distances = []
        prev = None
        for f_name in files:
            try:
                with open(os.path.join(data_dir, f_name), 'r') as f:
                    v = json.load(f)
                    if word in v:
                        curr = np.array(v[word])
                        if prev is not None: 
                            distances.append(np.linalg.norm(curr - prev))
                        prev = curr
            except:
                continue 

        plt.figure(figsize=(6, 3))
        plt.plot(distances, label=f"Stability of {word}")
        plt.title(f"Stability: '{word}' Embeddings\nWindow = {window}", fontsize=10)
        plt.legend(fontsize=8)
        
        plt.tight_layout()
        save_path = f"{out_dir}{word}_stability_window_{window}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Finished stability plot for '{word}': {save_path}")

if __name__ == "__main__":
    try:
        consolidated_path = ensure_consolidated_data(WANTED_WORDS, WINDOW_SIZE, SOURCE_DATA_PATH)
        
        if consolidated_path:
            print(f"Loading data from {consolidated_path}...")
            with open(consolidated_path, 'r', encoding='utf-8') as f:
                full_data = json.load(f)
            
            if full_data:
                if RUN_CONFIG["PCA_2D"]: plot_pca_2d(full_data, WANTED_WORDS, WINDOW_SIZE)
                if RUN_CONFIG["PCA_3D_POINTS"]: plot_pca_3d_points(full_data, WANTED_WORDS, WINDOW_SIZE)
                if RUN_CONFIG["PCA_3D_VECTORS"]: plot_pca_3d_vectors(SOURCE_DATA_PATH, WANTED_WORDS, WINDOW_SIZE, ITERATION_TO_PLOT)
                if RUN_CONFIG["UMAP_2D"]: plot_umap_2d(full_data, WANTED_WORDS, WINDOW_SIZE)
                if RUN_CONFIG["UMAP_3D"]: plot_umap_3d(full_data, WANTED_WORDS, WINDOW_SIZE)
                if RUN_CONFIG["STABILITY"]: plot_stability(SOURCE_DATA_PATH, WANTED_WORDS, WINDOW_SIZE)

        print("\nAll requested tasks completed.")
    except Exception as e:
        print(f"Error: {e}")
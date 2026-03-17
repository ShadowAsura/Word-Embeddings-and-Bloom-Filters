## Word Embeddings

This project explores different methods for generating word embeddings.

### Setup
1. Ensure Python 3.9 or higher is installed and the python3 alias points to such a version.
2. Set up a virtualenv.
```bash
pip install virtualenv
python3 -m virtualenv .venv
source .venv/bin/activate
```
3. And install the required packages.
```
pip install -r requirements.txt
```
### Structure

- `data/`: all files generated or used for data generation.
    - `fairy_tales/`: *(NOT COMMITTED)* binary .txt files for corpus
    - `text8/`: *(NOT COMMITTED)* optional text8 corpus (large raw text)
    - `indiv_word_representations/`: *(NOT COMMITTED)* generated using various representation methods, the vector for each instances of KQMW in the corpus. Generated in `indiv_word_representations.ipynb`.
    - `iterative_vectors/`: *(NOT COMMITTED)* stores vector representations across iterations, generated via `iterative_vectors.py`.
    - `kqmw_iterations/`: *(NOT COMMITTED)* measures how representations for KQMW generated using the iterative vectors method changes across many iterations. Generated in `iteration_data.ipynb`.
    - `fairytales_doc_tf-idf.json`: document-based TF-IDFs. Generated in `generate_tf-idfs_docuemnts.ipynb`. Not used.
    - `fairytales_tokenized.json`: tokenized + lemmatized corpus. Generated in `generate_tf-idfs_words.ipynb`.
    - `fairytales_word_bloom-filters.json`: *(NOT COMMITTED)* Bloom filter for words in corpus, generated in `generate_bloom_filters.ipynb`.
    - `kqmw_iteration.json`: *(NOT COMMITTED)* Generated in `iteration_data.ipynb` to consolidate representations of KQMW across iterations for use in PCA plots.
    - `n_neighbors.json`: *(NOT COMMITTED)* Helper file for use in `generate_tf-idfs_words.ipynb`.
    - `neighbor_frequencies.json`: *(NOT COMMITTED)* Helper file for use in `generate_tf-idfs_words.ipynb`.
    - `sentence_examples/`: examples of tokenized lemmatized sentences containing KQMW.
- `evaluation/`: evaluation scripts and question sets.
    - `analogy_questions_fairytales_clean.json`: cleaned semantic + syntactic analogy benchmark used for current diffusion vs Word2Vec comparisons.
- `eval/`: files from https://github.com/stanfordnlp/GloVe for analogies tests.
- `pca/`: directory for saving generated PCA plots. 
- `generate_bloom_filters.ipynb`: generates bloom filters for each word in the corpus.
- `generate_tf-idfs_documents.ipynb`: generates tf-idfs using the documents method, not used.
- `generate_tf-idfs_words.ipynb`: generates tf-idfs using the words method.
- `indiv_word_representations.ipynb`: given final representations for words in corpus, generates instance representations for specific words for further analysis.
- `iteration_data.ipynb`: generates data for analyzing representations across iterations generated via `iterative_vectors.py`.
- `pca_plots.ipynb`: generates PCA plots.
- `iterative_vectors.py`: generates vector representations using the iterative method.
- `results/analogies/`: committed CSV + Markdown outputs summarizing analogy evaluations (paper-style comparison table, benchmark cleanup notes, etc.).

### Full workflow (training → 10.7% analogy)

End-to-end order to build data, train, and evaluate to the reported ~10.7% total analogy accuracy (e.g. window 6 or 8 at iter 25).

**1. Corpus and tokenization**

- Put raw corpus text in `data/fairy_tales/` (e.g. `1.txt`, `2.txt`, …).
- In **`generate_tf-idfs_words.ipynb`**: build the tokenized corpus (and any helper files like `n_neighbors.json`, `neighbor_frequencies.json` if that notebook expects them), then save:
  - `data/fairytales_tokenized.json` (list of tokenized sentences).

**2. Word TF-IDF**

- In **`generate_tf-idfs_words.ipynb`**: compute word–word TF-IDFs from the tokenized corpus and save:
  - `data/fairytales_word_tf-idfs.json`.

**3. Bloom filters**

- Run **`generate_bloom_filters.ipynb`**: reads `data/fairytales_tokenized.json`, writes:
  - `data/fairytales_word_bloom-filters.json`.

**4. Initial vectors (iteration 0)**

- **`data/iterative_vectors/0.json`** must exist before training. It is a JSON `{ "word": [float, ...], ... }` with the same vocabulary as your TF-IDF/bloom data and one vector per word (e.g. from bloom filters or a prior run). Create it once (e.g. export bloom filters into this format into `data/iterative_vectors/0.json`) so the vocab and keys match the rest of the pipeline.

**5. Training (GPU)**

- From repo root:
  - Single run:  
    `NEIGHBORHOOD_SIZE=4 ALPHA=0.1 USE_ROBUST_SCALING=0 ITERATIONS=50 python iterative_vectors_v3.py`  
    Checkpoints are saved as `data/iterative_vectors/window_4_iter_1_v3_32bit.json`, …, `window_4_iter_50_v3_32bit.json`.
  - Or run the **window sweep** (windows 2, 4, 6, 8; ALPHA=0.1, USE_ROBUST_SCALING=0, 50 iters):  
    `python evaluation/run_window_sweep.py`  
    This trains each window, then evaluates checkpoints 1, 5, 10, 25, 50 and writes `evaluation/window_sweep_results.csv` and `evaluation/window_sweep_summary.txt`.

**6. Analogy evaluation**

- On a single checkpoint:  
  `python evaluation/evaluate_analogies.py --embeddings data/iterative_vectors/window_6_iter_25_v3_32bit.json`  
  You’ll see Semantic %, Syntactic %, and **Total %** (e.g. 10.7% for window 6 or 8 at iter 25).
- The **10.7%** number comes from this evaluator run on those checkpoints (produced by `iterative_vectors_v3.py`); it is not computed inside the training script.
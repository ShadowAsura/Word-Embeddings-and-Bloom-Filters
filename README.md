## Word Embeddings

This project explores multiple word-embedding pipelines, including diffusion-style
iterative vectors, Word2Vec baselines, and evaluation/reporting workflows used for
analogy and nearest-neighbor comparisons.

### Setup
1. Ensure Python 3.10 or higher is installed.
2. Create and activate a virtual environment.

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Important Git LFS note

Some large JSON artifacts are stored through Git LFS. If LFS objects are not pulled,
files may contain pointer text instead of real JSON and scripts can fail with
`JSONDecodeError`.

```bash
git lfs install
git lfs pull
```

### Structure

Status legend:
- `(COMMITTED)`: tracked in git history.
- `(NOT COMMITTED)`: local/generated/ignored files and folders.
- `(MIXED)`: contains both committed reference artifacts and local generated output.

- `README.md` `(COMMITTED)`: main project documentation.
- `requirements.txt` `(COMMITTED)`: Python dependency list.
- `.gitignore` `(COMMITTED)`: ignore policy for generated data/results artifacts.

- `fairy_tales/` `(COMMITTED)`: large numbered text corpus files used by the
  fairy-tale workflows.

- `data/` `(NOT COMMITTED)`: main generated-data workspace (ignored by default).
  - `data/text8/` `(NOT COMMITTED)`: optional text8 corpus assets.
  - `data/iterative_vectors/` `(NOT COMMITTED)`: diffusion checkpoints such as
    `window_4_iter_25_v3_32bit.json`.
  - `data/iterative_vectors_text8/` `(NOT COMMITTED)`: text8-specific iterative vectors.
  - `data/indiv_word_representations/` `(NOT COMMITTED)`: per-instance word vectors.
  - `data/word2vec/` `(NOT COMMITTED)`: Word2Vec exports/checkpoints.
  - `data/nnlm/` and `data/rnnlm/` `(NOT COMMITTED)`: language-model artifacts.
  - `data/text8_tokenized.json`, `data/text8_word_tf-idfs.json`,
    `data/text8_word_bloom-filters.json` `(NOT COMMITTED)`: generated tokenization,
    TF-IDF, and Bloom-filter data files.

- `evaluation/` `(MIXED)`: canonical active evaluation scripts and benchmark inputs.
  - `(COMMITTED)` benchmark/question assets:
    `evaluation/analogy_questions*.json`, `evaluation/questions-words.txt`.
  - `(COMMITTED)` active scripts:
    `evaluation/evaluate_analogies.py`, `evaluation/evaluate_google_analogies.py`,
    `evaluation/run_window_sweep.py`, `evaluation/run_alpha_sweep.py`,
    `evaluation/run_paper_eval.py`, `evaluation/plot_comparisons.py`,
    `evaluation/make_paper_table.py`, `evaluation/nearest_neighbors.py`,
    `evaluation/train_word2vec_baseline.py`, `evaluation/run_word2vec_eval.py`,
    `evaluation/run_diffusion_window_sweep.py`, and stability utilities.
  - `(NOT COMMITTED)` most newly generated `*.csv`, `*.txt`, and `*.png` run outputs.

- `results/` `(MIXED)`: canonical reporting/output area.
  - `results/plots/` `(MIXED)`: canonical location for generated plots and
    comparison figures.
  - `results/analogies/` `(COMMITTED)`: committed comparison scripts/tables/notes
    used for paper-style reporting.
  - `results/baselines/`, `results/logs/`, `results/stability/` `(NOT COMMITTED)`:
    local experiment outputs.

- `archive/legacy/` `(NOT COMMITTED)`: local archive for retired paths and plot outputs.
  - `archive/legacy/eval/` `(NOT COMMITTED)`: legacy evaluation scripts/notebooks.
  - `archive/legacy/plot_outputs/` `(NOT COMMITTED)`: migrated historical plots
    from older top-level output directories.

- `eval/` `(NOT PRESENT)`: retired top-level legacy evaluation path.
- `pca/` `(NOT PRESENT)`: retired top-level PCA output path.

- Top-level notebooks/scripts `(COMMITTED)`: exploratory generation and visualization
  notebooks plus training entry points, for example:
  - `generate_tf-idfs_words.ipynb`, `generate_tf-idfs_documents.ipynb`,
    `generate_bloom_filters.ipynb`, `indiv_word_representations.ipynb`,
    `iteration_data.ipynb`, `pca_plots.ipynb`, `pca_plots3D.ipynb`, `umap.ipynb`.
  - `iterative_vectors_v3.py`, `iterative_vectors.py`, `iterative_vectors8N.py`,
    `compare_vectors.py`, `plots.py`, `tokenize_text8.py`, `train_nnlm.py`,
    `train_rnnlm.py`.

### Full workflow (training to ~10.7% analogy)

End-to-end order to train and evaluate diffusion checkpoints that historically
reached around 10.7% total analogy accuracy (for example window 6 or 8 at iter 25).

**1. Corpus and tokenization**

- Put corpus files under `fairy_tales/` (or use `data/text8/` for text8 workflows).
- Build tokenized data in `generate_tf-idfs_words.ipynb`, then write the tokenized
  JSON used by downstream steps.

**2. Word TF-IDF**

- In `generate_tf-idfs_words.ipynb`, compute word-based TF-IDFs and save the
  resulting JSON file in `data/`.

**3. Bloom filters**

- Run `generate_bloom_filters.ipynb` to build word Bloom filters from tokenized data.

**4. Initial vectors (iteration 0)**

- Ensure an iteration-0 embedding JSON exists under `data/iterative_vectors/` with
  vocabulary aligned to your TF-IDF/Bloom inputs.

**5. Training (diffusion)**

- Single run example (PowerShell):

```powershell
$env:NEIGHBORHOOD_SIZE=4
$env:ALPHA=0.1
$env:USE_ROBUST_SCALING=0
$env:ITERATIONS=50
python iterative_vectors_v3.py
```

- Or run a multi-window sweep:

```bash
python evaluation/run_window_sweep.py
```

This writes sweep summaries such as `evaluation/window_sweep_results.csv` and
`evaluation/window_sweep_summary.txt`.

**6. Analogy evaluation**

- Evaluate one checkpoint with:

```bash
python evaluation/evaluate_analogies.py --embeddings data/iterative_vectors/window_6_iter_25_v3_32bit.json
```

- Run Google analogies format with:

```bash
python evaluation/evaluate_google_analogies.py \
  --embeddings data/iterative_vectors/window_4_iter_1_v3_32bit.json \
  --questions evaluation/questions-words.txt
```

### Common commands

Run from repository root.

```bash
# Diffusion sweep + evaluation CSV
python evaluation/run_diffusion_window_sweep.py

# Word2Vec evaluation CSV
python evaluation/run_word2vec_eval.py

# Comparison plots from evaluation CSVs
python evaluation/plot_comparisons.py

# Paper table + paper figures
python evaluation/make_paper_table.py --diffusion-csv evaluation/diffusion_window_sweep_results.csv

# Nearest-neighbor comparison report
python evaluation/nearest_neighbors.py
```

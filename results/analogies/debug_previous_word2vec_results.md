# Tracing earlier Word2Vec analogy numbers

## Goal

Explain where these earlier Word2Vec analogy numbers came from and why they differ from the **current** diffusion+Word2Vec comparison table:

- window 2: Semantic 20.0%, Syntactic 11.1%, Total 16.7%
- window 4: Semantic 26.7%, Syntactic 0.0%, Total 16.7%
- window 6: Semantic 20.0%, Syntactic 0.0%, Total 12.5%
- window 8: Semantic 20.0%, Syntactic 11.1%, Total 16.7%

## Files inspected

- `eval/word2vec_window_results.csv`
- `eval/run_word2vec_eval.py`
- `evaluation/analogy_questions.json`
- `results/analogies/paper_analogy_results.csv`
- `eval/diffusion_window_sweep_results.csv` (as a reference for other historical valid-counts)

## Where the earlier numbers came from

They were recorded in **`eval/word2vec_window_results.csv`** (Word2Vec windows 2/4/6/8, 32d) with **`total_valid = 24`**.

That CSV is produced by **`eval/run_word2vec_eval.py`**, which runs:

- `python evaluation/evaluate_analogies.py --embeddings data/word2vec/word2vec_vectors_32d_window{W}.json --questions evaluation/analogy_questions.json`

and parses the printed output into the CSV.

## Why they differ from the current table

**The valid-counts imply a different analogy set was used for the earlier results.**

Evidence:

- The earlier Word2Vec CSV rows have **`total_valid = 24`**.
- The current analogy set file **`evaluation/analogy_questions.json`** contains **20 semantic + 20 syntactic = 40** total questions.
- Current evaluation outputs (e.g., in `results/analogies/paper_analogy_results.csv`) show Word2Vec totals like **`21/40`** valid (semantic 20/20 + syntactic 1/20) on the present analogy set.

Conclusion:

- The earlier 24-valid Word2Vec numbers must have come from an **older/different** analogy question set than the current `evaluation/analogy_questions.json`, or from a different evaluation pipeline entirely.
- This repo currently contains **no alternate analogy file** besides `evaluation/analogy_questions.json`, so the earlier set is not reproducible from the current tree.

## Additional contributing mismatch (current setup)

On the current analogy set, Word2Vec has **only 1/20 syntactic valid** because the Word2Vec exported vocab (from the current `data/word2vec/*.json`) is missing:

- `went`, `took`, `gave`, `came`

Those four tokens appear in the current syntactic analogies, so most syntactic questions are skipped as OOV.

This explains why the current Word2Vec rows in `results/analogies/paper_analogy_results.csv` are **0.0%** with low syntactic validity: it’s a **vocab coverage mismatch** between the analogy set and the Word2Vec export.

## Most likely explanation (evidence-based)

The earlier nonzero Word2Vec numbers were obtained using an **older analogy set** (24 valid total) that:

- did **not** include the past-tense words that are currently missing from Word2Vec exports, and/or
- used different categories / questions than the current 20+20 set.

No copy of that older analogy set exists in the current repository, so the discrepancy is best explained as **analogy dataset drift** (questions changed) plus the known **Word2Vec OOV issue** on the present set.


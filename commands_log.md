## Script: train_nnlm.py
**Command:** `python train_nnlm.py --context 4 --dim 32 --hidden 128 --epochs 50 --batch 512 --device cuda --run-eval --questions evaluation/questions-words.txt`
**GPU:** NVIDIA GeForce RTX 5060, 8151 MiB
**Time:** 11.46 minutes
**Result:** Semantic 0.0%, Syntactic 0.1%, Total 0.1%

## Script: train_rnnlm.py
**Command:** `python train_rnnlm.py --rnn-type rnn --dim 32 --hidden 128 --epochs 50 --batch 64 --device cuda --run-eval --questions evaluation/questions-words.txt`
**GPU:** NVIDIA GeForce RTX 5060, 8151 MiB
**Time:** 50.94 minutes
**Result:** Semantic 0.4%, Syntactic 0.1%, Total 0.1%

## Script: iterative_vectors_v3.py (N=8)
**Command:** `$env:NEIGHBORHOOD_SIZE='8'; $env:ITERATIONS='150'; python iterative_vectors_v3.py`
**GPU:** NVIDIA GeForce RTX 5060, 8151 MiB
**Time:** 0.71 minutes
**Result:** Best checkpoint (from full Google eval sweep): Semantic 9.9%, Syntactic 0.5%, Total 1.3%

## Script: evaluation/evaluate_google_analogies.py (N=8 checkpoint sweep)
**Command:** `Get-ChildItem data/iterative_vectors/window_8_iter_*_v3_32bit.json | Sort-Object Name | ForEach-Object { python evaluation/evaluate_google_analogies.py --embeddings $_.FullName --questions evaluation/questions-words.txt }`
**GPU:** NVIDIA GeForce RTX 5060, 8151 MiB
**Time:** 0.10 minutes
**Result:** Best Semantic 9.9% (iter 1), Best Syntactic 0.5% (iter 4), Best Total 1.3% (iter 1)

## Script: iterative_vectors_v3.py (N=10)
**Command:** `$env:NEIGHBORHOOD_SIZE='10'; $env:ITERATIONS='150'; python iterative_vectors_v3.py`
**GPU:** NVIDIA GeForce RTX 5060, 8151 MiB
**Time:** 0.72 minutes
**Result:** Best checkpoint (from full Google eval sweep): Semantic 9.9%, Syntactic 0.7%, Total 1.4%

## Script: evaluation/evaluate_google_analogies.py (N=10 checkpoint sweep)
**Command:** `Get-ChildItem data/iterative_vectors/window_10_iter_*_v3_32bit.json | Sort-Object Name | ForEach-Object { python evaluation/evaluate_google_analogies.py --embeddings $_.FullName --questions evaluation/questions-words.txt }`
**GPU:** NVIDIA GeForce RTX 5060, 8151 MiB
**Time:** 0.09 minutes
**Result:** Best Semantic 9.9% (iter 1), Best Syntactic 0.7% (iter 4), Best Total 1.4% (iter 1)

## Script: evaluation/evaluate_google_analogies.py (Word2Vec 32d sweep)
**Command:** `Get-ChildItem data/word2vec/word2vec_vectors_32d_window*.json | Sort-Object Name | ForEach-Object { python evaluation/evaluate_google_analogies.py --embeddings $_.FullName --questions evaluation/questions-words.txt }`
**GPU:** NVIDIA GeForce RTX 5060, 8151 MiB
**Time:** 0.04 minutes
**Result:** Best Semantic 33.1% (window 2), Best Syntactic 2.5% (window 4), Best Total 6.5% (window 2)

## Script: evaluation/train_word2vec_baseline.py (CBOW 200d, window 2)
**Command:** `python evaluation/train_word2vec_baseline.py --window 2 --epochs 50 --negative 5 --dim 200 --sg 0 --out data/word2vec/word2vec_cbow_200d_window2.json`
**GPU:** NVIDIA GeForce RTX 5060, 8151 MiB (CPU only — gensim)
**Time:** < 1 minute
**Result:** Semantic 27.6%, Syntactic 3.3%, Total 6.8%

## Script: evaluation/train_word2vec_baseline.py (CBOW 200d, window 4)
**Command:** `python evaluation/train_word2vec_baseline.py --window 4 --epochs 50 --negative 5 --dim 200 --sg 0 --out data/word2vec/word2vec_cbow_200d_window4.json`
**GPU:** NVIDIA GeForce RTX 5060, 8151 MiB (CPU only — gensim)
**Time:** < 1 minute
**Result:** Semantic 27.9%, Syntactic 3.5%, Total 7.0%

## Script: evaluation/train_word2vec_baseline.py (CBOW 200d, window 6)
**Command:** `python evaluation/train_word2vec_baseline.py --window 6 --epochs 50 --negative 5 --dim 200 --sg 0 --out data/word2vec/word2vec_cbow_200d_window6.json`
**GPU:** NVIDIA GeForce RTX 5060, 8151 MiB (CPU only — gensim)
**Time:** < 1 minute
**Result:** Semantic 26.5%, Syntactic 3.7%, Total 7.0%

## Script: evaluation/train_word2vec_baseline.py (CBOW 200d, window 8)
**Command:** `python evaluation/train_word2vec_baseline.py --window 8 --epochs 50 --negative 5 --dim 200 --sg 0 --out data/word2vec/word2vec_cbow_200d_window8.json`
**GPU:** NVIDIA GeForce RTX 5060, 8151 MiB (CPU only — gensim)
**Time:** < 1 minute
**Result:** Semantic 22.8%, Syntactic 3.3%, Total 6.1%

## Script: train_rnnlm.py (padding bug fix applied)
**Fix:** Removed `padding_idx=0` from `nn.Embedding` constructor — index 0 was a real word, not PAD, so its embedding was frozen at zero.

## Script: evaluation/train_word2vec_baseline.py (200d, window 2)
**Command:** `python evaluation/train_word2vec_baseline.py --window 2 --epochs 50 --negative 5 --dim 200`
**GPU:** NVIDIA GeForce RTX 5060, 8151 MiB (CPU only — gensim)
**Time:** < 1 minute
**Result:** Semantic 10.7%, Syntactic 1.2%, Total 2.6%

## Script: evaluation/train_word2vec_baseline.py (200d, window 4)
**Command:** `python evaluation/train_word2vec_baseline.py --window 4 --epochs 50 --negative 5 --dim 200`
**GPU:** NVIDIA GeForce RTX 5060, 8151 MiB (CPU only — gensim)
**Time:** < 1 minute
**Result:** Semantic 12.1%, Syntactic 1.4%, Total 2.9%

## Script: evaluation/train_word2vec_baseline.py (200d, window 6)
**Command:** `python evaluation/train_word2vec_baseline.py --window 6 --epochs 50 --negative 5 --dim 200`
**GPU:** NVIDIA GeForce RTX 5060, 8151 MiB (CPU only — gensim)
**Time:** < 1 minute
**Result:** Semantic 14.0%, Syntactic 1.2%, Total 3.0%

## Script: evaluation/train_word2vec_baseline.py (200d, window 8)
**Command:** `python evaluation/train_word2vec_baseline.py --window 8 --epochs 50 --negative 5 --dim 200`
**GPU:** NVIDIA GeForce RTX 5060, 8151 MiB (CPU only — gensim)
**Time:** < 1 minute
**Result:** Semantic 18.8%, Syntactic 0.9%, Total 3.5%

## Script: train_nnlm.py (200d)
**Command:** `python train_nnlm.py --context 4 --dim 200 --hidden 128 --epochs 50 --batch 512 --device cuda`
**GPU:** NVIDIA GeForce RTX 5060, 8151 MiB
**Time:** 11.38 minutes (682.7s)
**Result:** Semantic 0.0%, Syntactic 0.0%, Total 0.0%

## Script: train_rnnlm.py (200d, padding bug fixed)
**Command:** `python train_rnnlm.py --rnn-type rnn --dim 200 --hidden 128 --epochs 50 --batch 64 --device cuda`
**GPU:** NVIDIA GeForce RTX 5060, 8151 MiB
**Time:** 44.23 minutes (2653.9s)
**Result:** Semantic 0.0%, Syntactic 0.0%, Total 0.0%

## Script: iterative_vectors_v3.py (N=8, PREROBUST_DUMP investigation)
**Command:** `NEIGHBORHOOD_SIZE=8 ITERATIONS=10 PREROBUST_DUMP=1 python iterative_vectors_v3.py`
**GPU:** NVIDIA GeForce RTX 5060, 8151 MiB
**Time:** 0.38 minutes (22.95s)
**Result (pre-robust vs post-robust):**
  - Iter 0 pre-robust: Semantic 5.9%, Syntactic 0.3%
  - Iter 0 post-robust: Semantic 7.7%, Syntactic 0.2%
  - Iter 1 pre-robust: Semantic 7.7%, Syntactic 0.2%
  - Iter 1 post-robust: Semantic 9.9%, Syntactic 0.1%  ← PEAK
  - Iter 2 pre-robust: Semantic 10.3%, Syntactic 0.6%  ← highest pre-robust
  - Iter 4 post-robust: Semantic 4.8%, Syntactic 0.5%
  - Iter 9 post-robust: Semantic 1.5%, Syntactic 0.1%
**Finding:** Robust scaling is NOT the primary cause of collapse. Pre-robust vectors also degrade over iterations. The robust step actually IMPROVES accuracy at iter 0 (5.9% → 7.7%) and iter 1 (7.7% → 9.9%). The diffusion process itself causes semantic collapse, likely due to over-smoothing.

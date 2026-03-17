# Diffusion vs Word2Vec analogy summary

| Method    | Config        | Best Iter | Semantic | Syntactic | Total | Semantic Valid/Total | Syntactic Valid/Total | Total Valid/Total |
|-----------|--------------|-----------|----------|-----------|-------|----------------------|-----------------------|-------------------|
| Diffusion | N=2          | 4         | 33.3%    | 0.0%      | 23.5% | 12/12                | 5/5                   | 17/17             |
| Diffusion | N=4          | 1         | 41.7%    | 0.0%      | 29.4% | 12/12                | 5/5                   | 17/17             |
| Diffusion | N=6          | 1         | 25.0%    | 0.0%      | 17.6% | 12/12                | 5/5                   | 17/17             |
| Diffusion | N=8          | 4         | 25.0%    | 0.0%      | 17.6% | 12/12                | 5/5                   | 17/17             |
| Word2Vec  | window=2     | -         | 91.7%    | 0.0%      | 64.7% | 12/12                | 5/5                   | 17/17             |
| Word2Vec  | window=4     | -         | 75.0%    | 0.0%      | 52.9% | 12/12                | 5/5                   | 17/17             |
| Word2Vec  | window=6     | -         | 75.0%    | 0.0%      | 52.9% | 12/12                | 5/5                   | 17/17             |
| Word2Vec  | window=8     | -         | 75.0%    | 0.0%      | 52.9% | 12/12                | 5/5                   | 17/17             |
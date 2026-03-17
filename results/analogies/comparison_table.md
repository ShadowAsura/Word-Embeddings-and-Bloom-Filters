# Diffusion vs Word2Vec analogy summary

| Method    | Config        | Best Iter | Semantic | Syntactic | Total | Semantic Valid/Total | Syntactic Valid/Total | Total Valid/Total |
|-----------|--------------|-----------|----------|-----------|-------|----------------------|-----------------------|-------------------|
| Diffusion | N=2          | 4         | 30.8%    | 0.0%      | 16.0% | 13/13                | 12/12                 | 25/25             |
| Diffusion | N=4          | 1         | 38.5%    | 0.0%      | 20.0% | 13/13                | 12/12                 | 25/25             |
| Diffusion | N=6          | 1         | 23.1%    | 0.0%      | 12.0% | 13/13                | 12/12                 | 25/25             |
| Diffusion | N=8          | 4         | 23.1%    | 0.0%      | 12.0% | 13/13                | 12/12                 | 25/25             |
| Word2Vec  | window=2     | -         | 84.6%    | 0.0%      | 44.0% | 13/13                | 12/12                 | 25/25             |
| Word2Vec  | window=4     | -         | 69.2%    | 0.0%      | 36.0% | 13/13                | 12/12                 | 25/25             |
| Word2Vec  | window=6     | -         | 69.2%    | 0.0%      | 36.0% | 13/13                | 12/12                 | 25/25             |
| Word2Vec  | window=8     | -         | 69.2%    | 0.0%      | 36.0% | 13/13                | 12/12                 | 25/25             |
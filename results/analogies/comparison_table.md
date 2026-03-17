# Diffusion vs Word2Vec analogy summary

| Method    | Config        | Best Iter | Semantic | Syntactic | Total | Semantic Valid/Total | Syntactic Valid/Total | Total Valid/Total |
|-----------|--------------|-----------|----------|-----------|-------|----------------------|-----------------------|-------------------|
| Diffusion | N=2          | 0         | 10.0%    | 0.0%      | 9.5%  | 20/20                | 1/1                   | 21/21             |
| Diffusion | N=4          | 1         | 15.0%    | 0.0%      | 14.3% | 20/20                | 1/1                   | 21/21             |
| Diffusion | N=6          | 0         | 10.0%    | 0.0%      | 9.5%  | 20/20                | 1/1                   | 21/21             |
| Diffusion | N=8          | 0         | 10.0%    | 0.0%      | 9.5%  | 20/20                | 1/1                   | 21/21             |
| Word2Vec  | window=2     | —         | 0.0%     | 0.0%      | 0.0%  | 20/20                | 1/1                   | 21/21             |
| Word2Vec  | window=4     | —         | 0.0%     | 0.0%      | 0.0%  | 20/20                | 1/1                   | 21/21             |
| Word2Vec  | window=6     | —         | 0.0%     | 0.0%      | 0.0%  | 20/20                | 1/1                   | 21/21             |
| Word2Vec  | window=8     | —         | 0.0%     | 0.0%      | 0.0%  | 20/20                | 1/1                   | 21/21             |
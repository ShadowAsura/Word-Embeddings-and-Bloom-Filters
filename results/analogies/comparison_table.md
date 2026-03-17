# Diffusion vs Word2Vec analogy summary

| Method    | Config        | Best Iter | Semantic | Syntactic | Total | Semantic Valid/Total | Syntactic Valid/Total | Total Valid/Total |
|-----------|--------------|-----------|----------|-----------|-------|----------------------|-----------------------|-------------------|
| Diffusion | N=2          | 0         | 10.0%    | 0.0%      | 5.0%  | 20/20                | 20/20                 | 40/40             |
| Diffusion | N=4          | 1         | 15.0%    | 0.0%      | 7.5%  | 20/20                | 20/20                 | 40/40             |
| Diffusion | N=6          | 0         | 10.0%    | 0.0%      | 5.0%  | 20/20                | 20/20                 | 40/40             |
| Diffusion | N=8          | 0         | 10.0%    | 0.0%      | 5.0%  | 20/20                | 20/20                 | 40/40             |
| Word2Vec  | window=2     | —         | 0.0%     | 0.0%      | 0.0%  | 20/20                | 1/20                  | 21/40             |
| Word2Vec  | window=4     | —         | 0.0%     | 0.0%      | 0.0%  | 20/20                | 1/20                  | 21/40             |
| Word2Vec  | window=6     | —         | 0.0%     | 0.0%      | 0.0%  | 20/20                | 1/20                  | 21/40             |
| Word2Vec  | window=8     | —         | 0.0%     | 0.0%      | 0.0%  | 20/20                | 1/20                  | 21/40             |
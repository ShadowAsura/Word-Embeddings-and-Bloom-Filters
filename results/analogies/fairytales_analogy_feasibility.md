# Fairytales analogy feasibility report (shared vocab)

## Shared vocabulary intersection
- Shared vocab size (intersection over 8 embeddings): **17620**

## Candidate semantic relation pairs (curated)
- Found **15** curated semantic pairs present in shared vocab.

- `king` ↔ `queen`
- `prince` ↔ `princess`
- `man` ↔ `woman`
- `boy` ↔ `girl`
- `father` ↔ `mother`
- `brother` ↔ `sister`
- `son` ↔ `daughter`
- `husband` ↔ `wife`
- `male` ↔ `female`
- `old` ↔ `young`
- `black` ↔ `white`
- `day` ↔ `night`
- `sun` ↔ `moon`
- `good` ↔ `bad`
- `big` ↔ `small`

## Candidate syntactic relation pairs (heuristic mining)
Counts below are the number of word pairs `(base, variant)` found in shared vocab for each relation.

| Relation | Pair count | Examples |
|----------|------------|----------|
| `irregular_past` | 6 | run->ran, see->saw, say->said, get->got, find->found |
| `plural_es` | 6 | bat->bates, dam->dames, gat->gates, goodby->goodbyes, jam->james |
| `plural_s` | 279 | adieu->adieus, adversary->adversarys, afghan->afghans, afterward->afterwards, ala->alas |
| `plural_y_ies` | 0 |  |
| `suffix_ed` | 261 | absorb->absorbed, accomplish->accomplished, accustom->accustomed, acquaint->acquainted, addict->addicted |
| `suffix_er` | 380 | add->adder, alt->alter, anoint->anointer, arch->archer, armor->armorer |
| `suffix_est` | 37 | bad->badest, c->cest, clever->cleverest, dark->darkest, dear->dearest |
| `suffix_ing` | 594 | absorb->absorbing, adjoin->adjoining, adorn->adorning, agree->agreeing, air->airing |

## Feasibility decision
- Minimum target to build a usable benchmark: **≥10 semantic analogies and ≥10 syntactic analogies**.
- This script only measures whether the *raw ingredients* exist (pairs), not whether they form clean non-duplicate analogy quadruples.

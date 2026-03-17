# Fairytales analogy benchmark cleanup

## Previous benchmark

- Source file: `evaluation/analogy_questions_fairytales.json`
- **Semantic questions**: 20
- **Syntactic questions**: 20

Issues:

1. **Syntactic section was highly repetitive**  
   - 20 questions were generated from only 6 irregular verb pairs (`run/ran`, `see/saw`, `say/said`, `get/got`, `find/found`, `think/thought`).  
   - Many questions were simple mirrors or permutations, e.g. both `run:ran::see:saw` and `see:saw::run:ran`, plus many near-duplicates with the same small set of verbs.

2. **Semantic section contained several weak or loosely-related analogies**  
   - Items like:
     - `["good", "big", "small", "bad"]`
     - `["water", "sea", "land", "fire"]`
     - `["old", "man", "woman", "young"]`
     - `["give", "take", "go", "come"]`  
   - These behave more like loose oppositions or associations than clean analogy relations, and are harder to defend in a paper.

## Cleaned benchmark

- New file: `evaluation/analogy_questions_fairytales_clean.json`
- **Semantic questions (kept)**: 16
  - Kept only clearly structured role/title/family analogies and directionally clear semantic relations:
    - `king/man :: queen/woman`
    - `prince/man :: princess/woman`
    - `father/man :: mother/woman`
    - `son/boy :: daughter/girl`
    - `brother/boy :: sister/girl`
    - `king/prince :: princess/queen`
    - `man/father :: mother/woman`
    - `boy/son :: daughter/girl`
    - `day/sun :: moon/night`
    - `black/night :: day/white`
    - `king/queen :: princess/prince`
    - `father/mother :: daughter/son`
    - `brother/sister :: girl/boy`
    - `big/small :: young/old`
    - `man/boy :: girl/woman`
    - `prince/king :: queen/princess`
- **Semantic questions removed (4)**:
  - `["good", "big", "small", "bad"]` – loose mix of size and valence; not a clear analogy pattern.
  - `["water", "sea", "land", "fire"]` – weak, ambiguous relation; closer to associative grouping.
  - `["old", "man", "woman", "young"]` – difficult to interpret as a consistent, symmetric relation.
  - `["give", "take", "go", "come"]` – more like paired opposites / motion verbs than a stable analogy.

- **Syntactic questions (kept)**: 12
  - Built from irregular past-tense pairs actually present in the shared vocab:
    - `run/ran`, `see/saw`, `say/said`, `get/got`, `find/found`, `think/thought`.
  - Chosen to avoid mirror duplicates and to encourage diversity:
    - `run:ran :: see:saw`
    - `run:ran :: say:said`
    - `run:ran :: get:got`
    - `run:ran :: find:found`
    - `run:ran :: think:thought`
    - `see:saw :: say:said`
    - `see:saw :: get:got`
    - `see:saw :: find:found`
    - `see:saw :: think:thought`
    - `say:said :: get:got`
    - `say:said :: find:found`
    - `say:said :: think:thought`

- **Syntactic questions removed (8)**:
  - All questions of the form:
    - `["see", "saw", "run", "ran"]` (mirror of `run:ran::see:saw`)
    - `["say", "said", "run", "ran"]`
    - `["say", "said", "see", "saw"]`
    - `["get", "got", "run", "ran"]`
    - `["get", "got", "see", "saw"]`
    - `["get", "got", "say", "said"]`
    - `["get", "got", "find", "found"]`
    - `["get", "got", "think", "thought"]`  
  - These were near-mirrors or added very little extra information over the kept set, given the same small verb inventory.
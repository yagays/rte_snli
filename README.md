# Recognizing Textual Entrailment with SNLI datasets
## Dataset

[The Stanford Natural Language Processing Group](https://nlp.stanford.edu/projects/snli/)

## Result

| Source | Model | Test Accuracy (% acc) |
|:-----------|------------:|:------------:|
| lstm.py | merge(200D LSTMs w/ pretrained word embedding) + MLP(300U*3L) | 80.5 |

D: Dimension, U: Unit, L: Layer

## Author

@yag_ays (<yanagi.ayase@gmail.com>)

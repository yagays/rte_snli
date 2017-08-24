# Recognizing Textual Entrailment with SNLI datasets
## Dataset

[The Stanford Natural Language Processing Group](https://nlp.stanford.edu/projects/snli/)

## Result

| Source | Model | Test Accuracy (% acc) |
|:-----------|------------:|:------------:|
| lstm.py | merge(200D LSTMs) + MLP(300U*3L) | 80.1 |
| lstm_td.py | merge(200D LSTMs + 100D TimeDistributed) + MLP(300U*3L) | 81.6 |
| lstm_bid.py | merge(200D BiDirectional LSTMs + 100D TimeDistributed) + MLP(200U*3L) | 82.1 |
| lstm_bid.py | merge(200D BiDirectional LSTMs + 200D TimeDistributed) + MLP(400U*3L) | 83.0 |

D: Dimension, U: Unit, L: Layer

## Author

@yag_ays (<yanagi.ayase@gmail.com>)

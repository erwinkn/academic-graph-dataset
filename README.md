# Challenge
Based on the [INF554 2021 Kaggle Challenge](https://www.kaggle.com/c/inf554-2021/)

# Run order
Data is expected in a `data` subfolder, with the same names as [provided in the Kaggle challenge](https://www.kaggle.com/c/inf554-2021/data).
If you don't want to install Rust to run the text processing script, I have provided the result in `processed/abstracts.txt`
- `processing.py` for engineered features
- `node2vec.py` for node2vec features
- `src/main.rs` for text processing
- `sbert.py` for text features
- `train.py` to train model and output predictions

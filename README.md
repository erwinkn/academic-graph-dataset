# Challenge
Predicting the h-index of academic authors from a subset of the Microsoft Academic Graph dataset. 
Based on the [INF554 2021 Kaggle Challenge](https://www.kaggle.com/c/inf554-2021/).

# Pipeline
The project sets up a hyperparameter sweep using the [Weights & Biases](https://wandb.ai/site) platform. The model is a [GraphSAGE](https://snap.stanford.edu/graphsage/) core, with 1 or 2 linear encoder or decoder layers, depending on the config. The encoder and decoder layers are important to wrap the GraphSAGE model, since the input size is very large (including at least 768 text features) and the output should be a single number (the h-index).

The initial data contains:
- A coauthorship edgelist
- A list of `(author_id, most_cited_papers)` where `most_cited_papers` is a list of 0 to 5 paper IDs
- A list of `(paper_id, abstract)` where the abstract is provided as an inverted index, as a dictionary of the form `{ word: positions_in_abstract[] }`

The extracted features are:
- Graph features like the degree, core decomposition or community-based centrality
- `node2vec` features, extracted using the unsupervised algorithm of the same name
- Text embeddings, produced by the following process:
    - For each paper, reconstruct its abstract from the inverted index (uses a Rust script for correctness and performance)
    - Use the [SPECTER](https://arxiv.org/abs/2004.07180) model (a [sentence transformer](https://www.sbert.net/index.html) pre-trained on scientific publications) to produce embeddings for each abstract
    - For each author, aggregate the embeddings of the abstract of their most cited papers by taking the mean (if available)

# Run order
Data is expected in a `data` subfolder, with the same names as [provided in the Kaggle challenge](https://www.kaggle.com/c/inf554-2021/data).
If you don't want to install Rust to run the text processing script, I have provided the result in `processed/abstracts.txt`
- `graph_features.py` for graph features
- `node2vec.py` for node2vec features
- `src/main.rs` for text processing
- `sbert.py` for text features
- `train.py` to train model and output predictions

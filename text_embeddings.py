def produce_text_embeddings():
    import pandas as pd
    import torch

    from sentence_transformers import SentenceTransformer
    from pathlib import Path

    abstracts = pd.read_table('processed/abstracts.txt', sep=',', header=None, index_col=0, names=['text'])
    path = Path('processed/abstract_embeddings.pt')
    if path.is_file():
        print('Found text_embeddings file! Loading...')
        encodings = torch.tensor(torch.load(path))
    else:
        print('Could not find abstract_embeddings file. Initiating encoding. This will take a while...')
        model = SentenceTransformer('allenai-specter', cache_folder='cache').cuda()
        encodings = model.encode(abstracts['text'].values, show_progress_bar=True)
        torch.save(encodings, 'processed/abstract_embeddings.pt')

    # Goal: aggregate the embeddings for each author and reorder them according to the node map
    author_papers = pd.read_table('data/author_papers.txt', sep="[:-]", index_col=0, header=None)
    # using the same semantics as the `processing` notebook and networkit
    node_map = pd.read_csv('processed/node_map.csv', index_col='author')
    author_map = node_map.reset_index().set_index('node')
    abstract_to_index = { id: index for index, id in enumerate(abstracts.index) }
    abstract_table = pd.DataFrame.from_dict(abstract_to_index, orient="index").astype(float)

    # Not all paper IDs have abstracts in our dataset
    # don't miss the - sign, to apply a boolean `not`
    no_abstract = - author_papers.isin(abstract_table.index)
    author_papers[no_abstract] = pd.NA
    author_papers = author_papers.applymap(lambda id: abstract_to_index[id], na_action='ignore')
    node_papers = author_papers.join(node_map)
    node_papers.set_index('node', inplace=True)
    node_papers.sort_index(inplace=True)

    # this is slow, but easy to write
    nb_nodes = node_papers.shape[0]
    i, row = next(node_papers.iterrows())
    node_text_embeddings = torch.zeros((nb_nodes, encodings.shape[1]))
    for i, row in node_papers.iterrows():
        row = row.dropna().to_numpy()
        ids = torch.tensor(row).long()
        if len(ids) == 0:
            embeddings = torch.zeros(encodings.size(1))
        else:
            embeddings = encodings[ids]
        node_text_embeddings[i] = embeddings.mean(dim=0)

    torch.save(node_text_embeddings, 'processed/text_embeddings.pt')
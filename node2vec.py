import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch_geometric.nn.models import Node2Vec

def produce_node_embeddings():
    # run processing.ipynb notebook to create / update the file
    data = torch.load('processed/data.pt')
    data.y = data.y.ravel()

    graph_size = data.x.size(0)
    # using the same semantics as the `processing` notebook and networkit
    node_map = pd.read_csv('processed/node_map.csv', index_col='author')
    author_map = node_map.reset_index().set_index('node')

    model = Node2Vec(
        data.edge_index,
        embedding_dim=128,
        walk_length=20,
        walks_per_node=10,
        context_size=10,
        num_negative_samples=1,
        p=0.25,
        q=0.25,
        sparse=True
    ).cuda()
    loader = model.loader(batch_size=1024, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.cuda(), neg_rw.cuda())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    EPOCHS = 50
    for epoch in range(1, EPOCHS+1):
        loss = train()
        print(f'Epoch: {epoch}, loss: {loss}')

    model.eval()
    embeddings = model().detach()
    embeddings = embeddings.cpu()
    torch.save(embeddings, "processed/node_embeddings.pt")
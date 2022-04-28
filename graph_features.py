import math
import networkit as nk
import numpy as np
from networkit.graphio import EdgeListReader
import pandas as pd
import torch

# stratified continuous split
from scs import scsplit
from numba import jit
from torch_geometric.data import Data

@jit(nopython=True)
def average_neighbor_degree(neighbors, degree):
    n = len(degree)
    result = np.zeros(len(degree))
    for v in range(n):
        total = 0
        ws = neighbors[v]
        for w in ws:
            total += degree[w]
        total /= len(ws)
        result[v] = total
    return result

# Metrics taken from "Can Author Collaboration Reveal Impact? The Case of h-index" (Nikolentzos et al., 2021)
# https://arxiv.org/abs/2104.05562

@jit(nopython=True)
def community_based_centrality(node_community_id, community_sizes, neighbors):
    n = len(node_community_id)
    centrality = np.zeros(n)
    for v in range(n):
        # for each neighbor, the centrality of v is augmented by
        # n_c / n, where n_c is the cardinality of the community of v
        for w in neighbors[v]:
            community_id = node_community_id[w]
            size = community_sizes[community_id]
            centrality[v] += size
    return centrality / n

@jit(nopython=True)
def community_mediator_centrality(degree, neighbors, nb_communities, node_community_id):
    n = len(degree)
    # reuse the same memory for each iteration
    densities = np.zeros(nb_communities)
    result = np.zeros(n)
    log_2 = math.log(2)

    for v in range(n):
        d = degree[v]
        # avoid diving by zero (result is 0 anyways)
        if d == 0:
            continue
        
        # Compute the density of each community in the neighbors of v
        neighbor_d = 0
        for w in neighbors[v]:
            community_id = node_community_id[w]
            densities[community_id] += 1
            neighbor_d += degree[w]

        # Compute and sum the entropy for all non-zero densities
        # Easier to write out by hand to avoid problems with NaN / -Inf values
        entropy = 0
        for h in densities:
            if h == 0:
                continue
            h /= d
            # base 2 logarithm
            h = -h * math.log(h) / log_2
            entropy += h

        # Convert the entropy into the community mediator centrality
        entropy = densities.sum()
        entropy = entropy * d / neighbor_d
        result[v] = entropy
        densities.fill(0)

    return result

def produce_graph_features():
    # Reading from an EdgeList with networkit is quite brittle, this is the only way that works
    reader = EdgeListReader(separator=' ', firstNode=0, continuous=False, directed=False)
    G = reader.read('data/coauthorship.edgelist')
    graph_size = G.numberOfNodes()
    # gotcha: keys are stored as strings in the NodeMap
    node_map = { int(author): node for author, node in reader.getNodeMap().items() }
    node_map = pd.DataFrame.from_dict(node_map, orient="index", columns=["node"])
    node_map.rename_axis("author", inplace=True)
    author_map = node_map.reset_index().set_index('node')

    # this is actually the degree (= unnormalized degree centrality)
    degree = nk.centrality.DegreeCentrality(G).run().scores()
    # The highest scores are in the range 0.01 - 0.04 and the lowest scores around 0.002
    # This approximation error is good enough while providing us with good performance
    betweenness_centrality = nk.centrality.KadabraBetweenness(G, err=0.0001, delta=0.001).run().scores()
    core_number = nk.centrality.CoreDecomposition(G).run().getPartition().getVector()
    page_rank = nk.centrality.PageRank(G).run().scores()

    partition = nk.community.detectCommunities(G, algo=nk.community.PLM(G, refine=True, turbo=True, gamma=1, recurse=True, maxIter=300))
    neighbors = [np.fromiter(G.iterNeighbors(n), dtype=int) for n in G.iterNodes()]
    node_community_id = partition.getVector()
    # apparently, some communities can be empty, so we need to cover the whole range
    # manually and not rely on partition.getSubsetIds
    nb_communities = max(node_community_id) + 1
    community_sizes = np.asarray([len(partition.getMembers(i)) for i in range(nb_communities)])
    node_community_size = [community_sizes[id] for id in node_community_id]

    # ignore the Numba warnings:
    # the library wants us to use the new numba.typed.List API, but it is much slower (for now)
    neighbor_degree = average_neighbor_degree(neighbors, degree)
    community_centrality = community_based_centrality(node_community_id, community_sizes, neighbors)
    mediator_centrality = community_mediator_centrality(degree, neighbors, len(community_sizes), node_community_id)

    abstracts = pd.read_table('data/author_papers.txt', sep="[:-]", index_col=0, header=None)
    abstracts = pd.DataFrame(abstracts.count(axis=1), columns=['papers_count'])
    abstracts.rename_axis("author", inplace=True)
    nb_papers = abstracts.join(node_map)
    nb_papers = nb_papers.set_index('node').sort_index()
    nb_papers = nb_papers['papers_count']

    def normal_tensor(input):
        tensor = torch.tensor(input).float()
        tensor = (tensor - tensor.mean()) / tensor.std()
        assert(not tensor.isnan().any()) # sanity check
        return tensor

    features = torch.stack([
        normal_tensor(degree),
        normal_tensor(betweenness_centrality),
        normal_tensor(core_number),
        normal_tensor(page_rank),
        normal_tensor(neighbor_degree),
        normal_tensor(community_centrality),
        normal_tensor(mediator_centrality),
        # TODO: experimenting with:
        # - community id
        # - community size
        # - papers count in author_papers.txt
        # normal_tensor(node_community_id),
        normal_tensor(node_community_size),
        normal_tensor(nb_papers)
    ], dim=1)
    # necessary for applying NN layers
    features = features.float()

    # Make the graph undirected
    edges = torch.tensor(list(G.iterEdges()))
    edges_rev = torch.stack([edges[:, 1], edges[:, 0]], dim=1)
    edge_index = torch.concat([edges, edges_rev], dim = 0).T

    # Train / val stratified continuous split + test mask
    test_csv = pd.read_csv('data/test.csv', usecols=['author'])
    test_nodes = node_map.loc[test_csv['author']]
    test_nodes = torch.tensor(test_nodes.values).flatten()
    test_mask = torch.zeros(graph_size).bool()
    test_mask[test_nodes] = True

    train_csv = pd.read_csv('data/train.csv')

    train_data, val_data = scsplit(
        train_csv,
        stratify=train_csv['hindex'],
        train_size=0.9,
        test_size=0.1
    )
    train_nodes = node_map.loc[train_data['author']]
    train_nodes = torch.tensor(train_nodes.values).flatten()
    train_mask = torch.zeros(graph_size).bool()
    train_mask[train_nodes] = True

    val_nodes = node_map.loc[val_data['author']]
    val_nodes = torch.tensor(val_nodes.values).flatten()
    val_mask = torch.zeros(graph_size).bool()
    val_mask[val_nodes] = True

    y = torch.zeros(graph_size)
    y[train_nodes] = torch.tensor(train_data['hindex'].values).float()
    y[val_nodes] = torch.tensor(val_data['hindex'].values).float()

    data = Data(
        x=features,
        edge_index=edge_index,
        y=y.view(-1, 1),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    torch.save(data, 'processed/data.pt')
    node_map.to_csv('processed/node_map.csv', index_label='author')
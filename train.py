# Prevent signal error
# https://github.com/wandb/client/issues/2854
import os
os.environ["WANDB_CONSOLE"] = "off"

import torch
import pandas as pd
import torch.nn.functional as F
import wandb

from argparse import ArgumentParser
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import LightningNodeData
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pathlib import Path

from text_embeddings import produce_text_embeddings
from graph_features import produce_graph_features
from node2vec import produce_node_embeddings
from model import Model
from sweep import sweep_config



def load_or_create(path, creator):
    path = Path(path)
    if not path.is_file():
        creator()
    return torch.load(path)

data = load_or_create('processed/data.pt', produce_graph_features)
# node_embeddings = load_or_create('processed/node_embeddings.pt', produce_node_embeddings)
text_embeddings = load_or_create('processed/text_embeddings.pt', produce_text_embeddings)
# change this line to include node_embeddings if needed
data.x = torch.cat([data.x, text_embeddings], dim=1)

graph_size = data.x.size(0)
nb_features = data.x.size(1)

def main(config):
    logger = WandbLogger()
    logger.log_hyperparams(config)

    datamodule = LightningNodeData(
        data,
        input_train_nodes=data.train_mask,
        input_val_nodes=data.val_mask,
        input_test_nodes=data.test_mask,
        loader="full"
    )
    model = Model(
        in_feats=nb_features,
        config=config
    )

    trainer = Trainer(
        max_epochs=5000,
        gpus=1,
        logger=logger,
        stochastic_weight_avg=config.stochastic_weight_avg,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=200)
        ]
    )

    trainer.fit(model, datamodule=datamodule)

wandb.init()
config = wandb.config

if __name__ == '__main__':
    main(config)


# --- SAVING ---
# using the same semantics as the `processing` notebook and networkit
# node_map = pd.read_csv('processed/node_map.csv', index_col='author')
# author_map = node_map.reset_index().set_index('node')
# authors = pd.read_csv('data/test.csv', usecols=['author'])
# authors = pd.Index(authors['author'])
# with torch.no_grad():
#     d = data.cuda()
#     out = model(d.x, d.edge_index).flatten().cpu()
#     preds = pd.DataFrame(out, columns=['hindex'])
#     preds = preds.rename_axis("node")
#     preds = preds.join(author_map)
#     preds = preds.set_index('author')
#     preds = preds.reindex(authors)
#     preds.to_csv('processed/predictions.csv')
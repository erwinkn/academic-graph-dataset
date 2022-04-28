import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn.models import GraphSAGE, GCN, GAT

import pytorch_lightning as pl
from torch_geometric.nn.models.basic_gnn import GIN


class Model(pl.LightningModule):
    def __init__(self, in_feats, config):
        super().__init__()

        self.lr = config.learning_rate
        self.dropout = config.dropout

        self.encoders = ModuleList()
        self.norms = ModuleList()

        self.encoders.append(Linear(in_feats, config.hidden_size))
        self.norms.append(BatchNorm1d(config.hidden_size))

        for i in range(1, config.encoders):
            encoder = Linear(config.hidden_size, config.hidden_size)
            norm = BatchNorm1d(config.hidden_size)
            self.encoders.append(encoder)
            self.norms.append(norm)

        if config.model == "SAGE":
            model_fn = GraphSAGE
        elif config.model == "GAT":
            model_fn = GAT
        else:
            model_fn = GCN

        model_out_size = config.hidden_size if config.decoders == 2 else 1

        self.model = model_fn(
            in_channels=config.hidden_size,
            hidden_channels=config.hidden_size,
            out_channels=model_out_size,
            num_layers=config.conv_layers,
            dropout=config.dropout,
            jk='cat',
            norm=BatchNorm1d(config.hidden_size)
        )

        if(config.decoders == 2):
            self.decoder_norm = BatchNorm1d(config.hidden_size)
            self.decoder = Linear(config.hidden_size, 1)

    # NOTE: here inference & training / validation are the same
    # Do not use forward in the training / validation steps if that's not the case
    def forward(self, data):
        x, edges = data.x, data.edge_index
        for i in range(0, len(self.encoders)):
            x = self.encoders[i](x)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)

        x = self.model(x, edges)

        if(hasattr(self, 'decoder')):
            x = self.decoder_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
            x = self.decoder(x)

        return x

    def training_step(self, batch, batch_idx):
        z = self.forward(batch)
        mask = batch.train_mask
        loss = F.mse_loss(z[mask], batch.y[mask])
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        z = self.forward(batch)
        mask = batch.val_mask
        loss = F.mse_loss(z[mask], batch.y[mask])
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
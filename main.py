import os
from stim_dataset import STIMDataset
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
from zigzag import peak_valley_pivots_candlestick
import pandas as pd
import pandas_ta as ta
from torch.nn import functional as F
import copy 

from torch.nn import Conv2d, MaxPool2d, Dropout, Flatten, Linear, Sequential, ReLU
from torch.nn.functional import dropout

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.classification import ConfusionMatrix

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Sequential(
            Conv2d(1, 32, (3, 3)),
            ReLU(),
            Conv2d(32, 64, (3, 3)),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
            Dropout(.25)
        )

        dummy_input = torch.zeros((1, 1, 14, 15))
        dummy_output = self.features(dummy_input)
        feature_size = dummy_output.shape[-1]
        
        self.dropout = Dropout(.5)
        self.pred_layer = Linear(feature_size, 3)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        return self.pred_layer(x)

class Ctok(pl.LightningModule):
    def __init__(self, model, dataset, batch_size = 16, learning_rate = 1e-3):
        super().__init__()
        self.hparams = {"batch_size": batch_size, "learning_rate": learning_rate}
        self.dataset_train = dataset
        self.dataset_val = copy.deepcopy(dataset)
        self.dataset_val.transform_to_val()
        self.model = model

        self.cf = ConfusionMatrix(3)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss, on_step=True)
        _, preds = y_hat.max(1)
        self.cf.update(preds, y)

    def validation_epoch_end(self, outputs):
        # Do something here
        print(self.cf.compute())
        self.dataset_train.increase_year()
        self.dataset_val.increase_year()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)


    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.batch_size,
        )

if __name__ == "__main__":
    model = Model()
    dataset = STIMDataset()
    engine = Ctok(model=model, dataset=dataset)
    tb_logger = TensorBoardLogger(save_dir="logs")
    trainer = Trainer(
        gpus=1,
        logger=tb_logger,
        benchmark=True,
        num_sanity_val_steps=0,
        reload_dataloaders_every_epoch=True,
        accelerator="dp"
    )
    trainer.fit(engine)
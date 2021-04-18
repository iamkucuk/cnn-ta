from data_analyzer import LabelHistogram
import pytorch_lightning as pl
import copy
from pytorch_lightning.metrics.classification.confusion_matrix import ConfusionMatrix
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from STIMDataset import *
from stats import calculate_stats


class Ctok(pl.LightningModule):
    def __init__(self, model, dataset, batch_size = 16, learning_rate = 1e-3):
        super().__init__()
        self.hparams = {"batch_size": batch_size, "learning_rate": learning_rate}
        self.dataset_train = dataset
        self.dataset_val = copy.deepcopy(dataset)
        self.dataset_val.transform_to_val()
        print(len(self.dataset_train))
        print(len(self.dataset_val))
        self.model = model
        self.cf = ConfusionMatrix(3, normalize="true")
        self.increase_every = None
        self.hist_training = LabelHistogram(["Hold", "Buy", "Sell"])
        self.hist_validation = LabelHistogram(["Hold", "Buy", "Sell"])
        self.hist_test = LabelHistogram(["Hold", "Buy", "Sell"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.hist_training.update(y.cpu().data.numpy())
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)
        self.hist_validation.update(y.cpu().data.numpy())
        _, preds = y_hat.max(1)
        self.cf.update(preds, y)
        return loss

    def validation_epoch_end(self, outputs):
        if self.increase_every is None:
            self.increase_every = np.ceil(self.trainer.max_epochs / (self.dataset_val.year_test - self.dataset_val.year_finish))
        hebe = self.cf.compute()
        self.cf.reset()
        print(hebe)
        p = torch.diagonal(hebe).sum()
        self.log('p', p)
        print(p)
        if (self.current_epoch + 1) % self.increase_every == 0:
            print("Checkpoint reached. Increasing this fucker...")
            self.dataset_train.increase_year()
            self.dataset_val.increase_year()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        _, preds = y_hat.max(1)
        self.cf.update(preds, y)
        self.hist_test.update(y.cpu().data.numpy())
        return preds

    def test_epoch_end(self, outputs):
        outputs = torch.stack(outputs).squeeze(1)
        self.test_dataset = self.test_dataloader.dataloader.dataset.tickers.dropna().loc[f"{self.test_dataloader.dataloader.dataset.year_test}-1-1":]
        self.test_signals = pd.Series(outputs.cpu().data.numpy())
        self.test_signals.index = self.test_dataset.adj_close.index
        self.stats = calculate_stats(self.test_dataset.adj_close, self.test_signals, 0.001)
        hebe = self.cf.compute()
        self.cf.reset()
        print(hebe)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adadelta(self.parameters(), lr=self.hparams.learning_rate, rho=.95, eps=1e-7)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.hparams.batch_size,
            sampler=ImbalancedDatasetSampler(self.dataset_train, callback_get_label=STIMDataset.STIMDatasetLabelCallback),
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.hparams.batch_size,
            num_workers=2,
        )
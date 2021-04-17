def original_label_strategy(close, window_size=11):
    """
    Data is labeled as per the logic in research paper
    Label code : BUY => 1, SELL => 2, HOLD => 0
    params :
        df => Dataframe with data
        col_name => name of column which should be used to determine strategy
    returns : numpy array with integer codes for labels with
                size = total-(window_size)+1
    """

    row_counter = 0
    total_rows = len(close)
    labels = np.zeros(total_rows)
    labels[:] = np.nan
    pbar = tqdm(total=total_rows, desc="Creating labels")

    while row_counter < total_rows:
        if row_counter >= window_size - 1:
            window_begin = row_counter - (window_size - 1)
            window_end = row_counter
            window_middle = (window_begin + window_end) // 2

            min_ = np.inf
            min_index = -1
            max_ = -np.inf
            max_index = -1
            for i in range(window_begin, window_end + 1):
                price = close[i]
                if price < min_:
                    min_ = price
                    min_index = i
                if price > max_:
                    max_ = price
                    max_index = i

            if max_index == window_middle:
                labels[window_middle] = 2
            elif min_index == window_middle:
                labels[window_middle] = 1
            else:
                labels[window_middle] = 0

        row_counter = row_counter + 1
        pbar.update(1)

    pbar.close()
    return labels

def _identify_initial_pivot(X, up_thresh, down_thresh):
    """Quickly identify the X[0] as a peak or valley."""
    PEAK, VALLEY = 1, -1

    x_0 = X[0]
    max_x = x_0
    max_t = 0
    min_x = x_0
    min_t = 0
    up_thresh += 1
    down_thresh += 1

    for t in range(1, len(X)):
        x_t = X[t]

        if x_t / min_x >= up_thresh:
            return VALLEY if min_t == 0 else PEAK

        if x_t / max_x <= down_thresh:
            return PEAK if max_t == 0 else VALLEY

        if x_t > max_x:
            max_x = x_t
            max_t = t

        if x_t < min_x:
            min_x = x_t
            min_t = t

    t_n = len(X)-1
    return VALLEY if x_0 < X[t_n] else PEAK


def peak_valley_pivots_candlestick(close, up_thresh=.1, down_thresh=-.1):#, high, low, up_thresh, down_thresh):
    """
    Finds the peaks and valleys of a series of HLC (open is not necessary).
    TR: This is modified peak_valley_pivots function in order to find peaks and valleys for OHLC.
    Parameters
    ----------
    close : This is series with closes prices.
    high : This is series with highs  prices.
    low : This is series with lows prices.
    up_thresh : The minimum relative change necessary to define a peak.
    down_thesh : The minimum relative change necessary to define a valley.
    Returns
    -------
    an array with 0 indicating no pivot and -1 and 1 indicating valley and peak
    respectively
    Using Pandas
    ------------
    For the most part, close, high and low may be a pandas series. However, the index must
    either be [0,n) or a DateTimeIndex. Why? This function does X[t] to access
    each element where t is in [0,n).
    The First and Last Elements
    ---------------------------
    The first and last elements are guaranteed to be annotated as peak or
    valley even if the segments formed do not have the necessary relative
    changes. This is a tradeoff between technical correctness and the
    propensity to make mistakes in data analysis. The possible mistake is
    ignoring data outside the fully realized segments, which may bias analysis.
    """
    if down_thresh > 0:
        raise ValueError('The down_thresh must be negative.')

    initial_pivot = _identify_initial_pivot(close, up_thresh, down_thresh)

    t_n = len(close)
    pivots = np.zeros(t_n, dtype='i1')
    pivots[0] = initial_pivot

    # Adding one to the relative change thresholds saves operations. Instead
    # of computing relative change at each point as x_j / x_i - 1, it is
    # computed as x_j / x_1. Then, this value is compared to the threshold + 1.
    # This saves (t_n - 1) subtractions.
    up_thresh += 1
    down_thresh += 1

    trend = -initial_pivot
    last_pivot_t = 0
    last_pivot_x = close[0]
    for t in tqdm(range(1, len(close)), "Creating labels"):

        if trend == -1:
            # x = low[t]
            x = close[t]
            r = x / last_pivot_x
            if r >= up_thresh:
                pivots[last_pivot_t] = trend
                trend = 1
                last_pivot_x = x
                last_pivot_t = t
            elif x < last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t
        else:
            # x = high[t]
            x = close[t]
            r = x / last_pivot_x
            if r <= down_thresh:
                pivots[last_pivot_t] = trend
                trend = -1
                last_pivot_x = x
                last_pivot_t = t
            elif x > last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t


    if last_pivot_t == t_n-1:
        pivots[last_pivot_t] = trend
    elif pivots[t_n-1] == 0:
        pivots[t_n-1] = trend

    pivots[pivots == 1] = 2
    pivots[pivots == -1] = 1

    return pivots

calc_ann_sharpe = lambda daily_returns: daily_returns.mean()*252 / (daily_returns.std()*np.sqrt(252))
calc_drawdown = lambda cum_returns: np.ptp(cum_returns)/cum_returns.max()

def calculate_stats(closes, signals, fee: float = 0):
    """Signals: 0 = hold, 1 = buy, 2 = sell"""
    daily_returns = closes.pct_change().iloc[1:]
    cum_returns = (daily_returns + 1).cumprod()
    annualized_sharpe_ratios = calc_ann_sharpe(daily_returns)#daily_returns.mean()*252 / (daily_returns.std()*np.sqrt(252))
    drawdown = calc_drawdown(cum_returns)#np.ptp(cum_returns)/cum_returns.max()

    signals_altered = signals.iloc[1:].replace(0, np.nan).replace(2, 0).ffill()
    trade_daily_returns = daily_returns * signals_altered
    trade_cum_returns = (trade_daily_returns + 1).cumprod()
    annualized_trade_sharpe_ratios = calc_ann_sharpe(trade_daily_returns)#trade_daily_returns.mean()*252 / (trade_daily_returns.std()*np.sqrt(252))
    trade_drawdown = calc_drawdown(trade_cum_returns)#np.ptp(trade_cum_returns)/trade_cum_returns.max()

    df = pd.concat([daily_returns, cum_returns, trade_daily_returns, trade_cum_returns], axis=1)
    df = df.set_axis(["daily_returns", "cum_returns", "trade_daily_returns", "trade_cum_returns"], axis="columns")
    return {
        "returns": df,
        "original_sharpe": annualized_sharpe_ratios,
        "original_drawdown": drawdown,
        "trade_sharpe": annualized_trade_sharpe_ratios,
        "trade_drawdown": trade_drawdown
        }
    

from typing import Tuple
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import pandas_ta as ta
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from enum import Enum
import torch
import numpy as np
import matplotlib.pyplot as plt

class LabellingStrategy(Enum):
    original = original_label_strategy
    zigzag = peak_valley_pivots_candlestick

class TrainingStage(Enum):
    training = 0
    validation = 1
    test = 2

class STIMDataset(Dataset):
    def __init__(self, csv_file="SPY_ta.csv", interval = range(6, 21), test_split_last_n = 7, label_method: LabellingStrategy = LabellingStrategy.original, strategy_kwargs: dict = {}, years_interval: Tuple = None, training: bool = True, test: bool = False) -> None:
        super().__init__()
        self.tickers = pd.read_csv(csv_file)
        self.tickers["Date"] = pd.to_datetime(self.tickers["Date"], format="%Y-%m-%d")
        self.tickers.set_index("Date", inplace=True)
        self.tickers.ta.adjusted = "adj_close"

        # self.num_years = self.tickers.index[-1].year - self.tickers.index[0].year
        
        self.indicators = ["rsi", "willr", "wma", "ema", "sma", "hma", "tema", 
                        "cci", "cmo", "macd", "ppo", "roc", "cmf", "adx"]#, "psar"]

        if len(self.tickers.columns) < 8:
            # if label_method == LabellingStrategy.original:
            self.tickers["labels"] = label_method(self.tickers["adj_close"], **strategy_kwargs)

            self._generate_tas(self.tickers, interval=interval)
        
        self.feature_groups_cols = [self.tickers.columns[7 + i * len(interval) : 7 + (i + 1) * len(interval)] for i in range(len(self.indicators))]

        self.normalized_dataset = self.tickers[self.tickers.columns]
        self.normalized_dataset.dropna(inplace=True)
        self.normalized_dataset = ((self.normalized_dataset - self.normalized_dataset.min()) / (self.normalized_dataset.max() - self.normalized_dataset.min())) * 2 - 1
        self.normalized_dataset["labels"] = self.normalized_dataset["labels"] + 1

        self.year_last = self.normalized_dataset.index[-1].year
        self.year_test = self.year_last - test_split_last_n

        self.normalized_dataset_test = self.normalized_dataset.loc[f"{self.year_test}-1-1":]
        self.normalized_dataset = self.normalized_dataset.loc[:f"{self.year_test}-1-1"]

        self.num_years = self.normalized_dataset.index[-1].year - self.normalized_dataset.index[0].year

        if years_interval is None:
            self.year_start = self.normalized_dataset.index[0].year
            self.year_finish = self.year_start + 4
        else:
            self.year_start, self.year_finish = years_interval
        self.year_val = self.year_finish + 1

        self.training = training
        if self.training:
            self.dataset_to_load = self.normalized_dataset[f"{self.year_start}-1-1":f"{self.year_finish}-1-1"]
        else:
            self.dataset_to_load = self.normalized_dataset[f"{self.year_val}-1-1":f"{self.year_val+1}-1-1"]

    def __len__(self):
        return len(self.dataset_to_load)

    def transform_to_val(self):
        self.training = False
        self.dataset_to_load = self.normalized_dataset[f"{self.year_val}-1-1":f"{self.year_val+1}-1-1"]

    def transform_to_test(self):
        self.dataset_to_load = self.normalized_dataset_test

    def increase_year(self):
        self.year_start +=1
        self.year_finish +=1
        self.year_val +=1

        if self.training:
            self.dataset_to_load = self.normalized_dataset[f"{self.year_start}-1-1":f"{self.year_finish}-1-1"]
        else:
            self.dataset_to_load = self.normalized_dataset[f"{self.year_val}-1-1":f"{self.year_val+1}-1-1"]

    def __getitem__(self, index):
        img = np.zeros((len(self.feature_groups_cols), len(self.feature_groups_cols[0])))
        for idx, cols in enumerate(self.feature_groups_cols):
            img[idx, :] = self.dataset_to_load.iloc[index][cols].values

        img = torch.tensor(img).unsqueeze(0).float()

        target = torch.tensor(self.dataset_to_load.iloc[index]["labels"]).long()
        # target[int(self.normalized_dataset.iloc[index]["labels"])] = 1

        return img, target
        

    def _generate_tas(self, df, interval=range(6, 21)):

        for indicator in tqdm(self.indicators, desc="Creating indicators"):
            indicator_method = getattr(df.ta, indicator)
            for length in interval:
                found = indicator_method(length=length, slow=length, fast=length+10)
                if isinstance(found, pd.DataFrame):
                    cols = found.columns
                    for col in cols:
                        col_cont = col[len(indicator):]
                        if col_cont[0] == "_":
                            found = found[col]
                            break
                
                df[found.name] = found

    def show_labeled_data(self):
        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
        self.tickers.plot(y="adj_close", use_index=True, ax=ax)
        plt.plot(self.tickers.adj_close[self.tickers.labels == 1], color="g", marker="*", linestyle="None")
        plt.plot(self.tickers.adj_close[self.tickers.labels == 2], color="r", marker="*", linestyle="None")

# dataset = STIMDataset()
# dataset.show_labeled_data()

import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import pandas as pd
from torch.nn import functional as F
import copy 

from torch.nn import Conv2d, MaxPool2d, Dropout, Flatten, Linear, Sequential, ReLU
from torch.nn.functional import dropout

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics.classification import ConfusionMatrix

import torch
import torch.utils.data
import torchvision


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

def STIMDatasetLabelCallback(dataset, idx):
    return dataset.__getitem__(idx)[1].cpu().data.item()

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
        print(len(self.dataset_train))
        print(len(self.dataset_val))
        self.model = model
        self.cf = ConfusionMatrix(3, normalize="true")
        self.increase_every = None

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
        self.log('valid_loss', loss)
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
        return preds

    def test_epoch_end(self, outputs):
        outputs = torch.stack(outputs).squeeze(1)
        closes = self.test_dataloader.dataloader.dataset.tickers.dropna().loc[f"{self.test_dataloader.dataloader.dataset.year_test}-1-1":].adj_close
        signals = pd.Series(outputs.cpu().data.numpy())
        signals.index = closes.index
        self.stats = calculate_stats(closes, signals, 0.001)
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
            sampler=ImbalancedDatasetSampler(self.dataset_train, callback_get_label=STIMDatasetLabelCallback),
            # shuffle=True,
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.hparams.batch_size,
            # shuffle=False,
            num_workers=2,
        )

if __name__ == "__main__":
    import wandb
    wandb.login()

    model = Model()
    dataset = STIMDataset(label_method=LabellingStrategy.zigzag, strategy_kwargs={"up_thresh":.02, "down_thresh":-.02})
    engine = Ctok(model=model, dataset=dataset, learning_rate=1, batch_size=256)
    engine.load_state_dict(torch.load("checkpoints\cnnta-epoch=26-p=2.64.ckpt")["state_dict"])
    tb_logger = WandbLogger(project="cnn-ta")
    checkpointer = ModelCheckpoint(
        monitor='p',
        mode="max",
        verbose=True,
        dirpath='checkpoints',
        filename='cnnta-{epoch:02d}-{p:.2f}',
        save_last=True,
        save_top_k=10,
    )
    trainer = Trainer(
        gpus=1,
        logger=tb_logger,
        benchmark=True,
        num_sanity_val_steps=1,
        reload_dataloaders_every_epoch=True,
        max_epochs=70,
        default_root_dir="checkpoints",
        resume_from_checkpoint="checkpoints\cnnta-epoch=26-p=2.64.ckpt",
        callbacks=[checkpointer]
    )

    # trainer.fit(engine)

    # engine = Ctok.load_from_checkpoint(checkpoint_path="checkpoints\cnnta-epoch=46-p=2.15.ckpt")
    dataset_test = STIMDataset()#label_method=LabellingStrategy.zigzag, strategy_kwargs={"up_thresh":.05, "down_thresh":-.05})
    dataset_test.transform_to_test()
    test_loader = DataLoader(dataset=dataset_test, batch_size=1)
    trainer.test(engine, test_loader)
    hebe = 0

# dataset_test = STIMDataset()#label_method=LabellingStrategy.zigzag, strategy_kwargs={"up_thresh":.05, "down_thresh":-.05})
# dataset_test.transform_to_test()
# test_loader = DataLoader(dataset=dataset_test, batch_size=1)
# trainer.test(engine, test_loader)

# dataset1 = STIMDataset()
# dataset1.transform_to_test()
# dataset2 = STIMDataset(label_method=LabellingStrategy.zigzag, strategy_kwargs={"up_thresh":.05, "down_thresh":-.05})
# dataset2.transform_to_test()



# dataset.show_labeled_data()
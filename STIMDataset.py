from balanced_sampler import ImbalancedDatasetSampler
from labellers import original_label_strategy, peak_valley_pivots_candlestick
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
    def __init__(self, csv_file="/content/src/SPY.csv", interval = range(6, 21), test_split_last_n = 7, label_method: LabellingStrategy = LabellingStrategy.original, strategy_kwargs: dict = {}, years_interval: Tuple = None, training: bool = True, test: bool = False) -> None:
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
        self.year_val = self.year_finish

        self.training = training
        self.sampler = None
        if self.training:
            self.dataset_to_load = self.normalized_dataset[f"{self.year_start}-1-1":f"{self.year_finish}-1-1"]
            self.sampler = ImbalancedDatasetSampler(self, callback_get_label=STIMDataset.STIMDatasetLabelCallback)
        else:
            self.dataset_to_load = self.normalized_dataset[f"{self.year_val}-1-1":f"{self.year_val+1}-1-1"]

    def __len__(self):
        return len(self.dataset_to_load)

    @staticmethod
    def STIMDatasetLabelCallback(dataset, idx):
        return dataset.__getitem__(idx)[1].cpu().data.item()

    def transform_to_val(self):
        self.training = False
        self.dataset_to_load = self.normalized_dataset[f"{self.year_val}-1-1":f"{self.year_val+1}-1-1"]
        self.sampler = None

    def transform_to_test(self):
        self.dataset_to_load = self.normalized_dataset_test
        self.sampler = None

    def increase_year(self):
        self.year_start +=1
        self.year_finish +=1
        self.year_val +=1

        if self.training:
            self.dataset_to_load = self.normalized_dataset[f"{self.year_start}-1-1":f"{self.year_finish}-1-1"]
        else:
            self.dataset_to_load = self.normalized_dataset[f"{self.year_val}-1-1":f"{self.year_val+1}-1-1"]

        print(f"New start year is: {self.dataset_to_load.index[0]}")
        print(f"New end year is: {self.dataset_to_load.index[-1]}")

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


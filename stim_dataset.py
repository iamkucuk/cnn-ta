from typing import Tuple
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import pandas_ta as ta
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from enum import Enum
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np

class LabellingStrategy(Enum):
    original = 0
    zigzag = 1

class STIMDataset(Dataset):
    def __init__(self, csv_file="SPY_ta.csv", interval = range(6, 21), label_method: LabellingStrategy = LabellingStrategy.original, years_interval: Tuple = None, training: bool = False) -> None:
        super().__init__()
        self.tickers = pd.read_csv(csv_file)
        self.tickers["Date"] = pd.to_datetime(self.tickers["Date"], format="%Y-%m-%d")
        self.tickers.set_index("Date", inplace=True)
        self.tickers.ta.adjusted = "adj_close"
        
        self.indicators = ["rsi", "willr", "wma", "ema", "sma", "hma", "tema", 
                        "cci", "cmo", "macd", "ppo", "roc", "cmf", "adx"]#, "psar"]

        if len(self.tickers.columns) < 8:
            if label_method == LabellingStrategy.original:
                self.tickers["labels"] = self.create_labels(self.tickers, "adj_close")
            else:
                raise NotImplementedError

            self._generate_tas(self.tickers, interval=interval)
        
        self.feature_groups_cols = [self.tickers.columns[7 + i * len(interval) : 7 + (i + 1) * len(interval)] for i in range(len(self.indicators))]

        self.normalized_dataset = self.tickers[self.tickers.columns]
        self.normalized_dataset.dropna(inplace=True)
        self.normalized_dataset = ((self.normalized_dataset - self.normalized_dataset.min()) / (self.normalized_dataset.max() - self.normalized_dataset.min())) * 2 - 1
        self.normalized_dataset["labels"] = self.normalized_dataset["labels"] + 1

        if years_interval is None:
            self.year_start = self.normalized_dataset.index[0].year
            self.year_finish = self.year_start + 5
        else:
            self.year_start, self.year_finish = years_interval
        self.year_test = self.year_finish + 1

        self.training = training
        if self.training:
            self.dataset_to_load = self.normalized_dataset[f"{self.year_start}-1-1":f"{self.year_finish}-1-1"]
        else:
            self.dataset_to_load = self.normalized_dataset[f"{self.year_test}-1-1":f"{self.year_test+1}-1-1"]

    def __len__(self):
        return len(self.dataset_to_load)

    def transform_to_val(self):
        self.training = False
        self.dataset_to_load = self.normalized_dataset[f"{self.year_test}-1-1":f"{self.year_test+1}-1-1"]

    def increase_year(self):
        self.year_start +=1
        self.year_finish +=1
        self.year_test +=1

        if self.training:
            self.dataset_to_load = self.normalized_dataset[f"{self.year_start}-1-1":f"{self.year_finish}-1-1"]
        else:
            self.dataset_to_load = self.normalized_dataset[f"{self.year_test}-1-1":f"{self.year_test+1}-1-1"]

            

    def __getitem__(self, index):
        img = np.zeros((len(self.feature_groups_cols), len(self.feature_groups_cols[0])))
        for idx, cols in enumerate(self.feature_groups_cols):
            img[idx, :] = self.normalized_dataset.iloc[index][cols].values

        img = torch.tensor(img).unsqueeze(0).float()

        target = torch.tensor(self.normalized_dataset.iloc[index]["labels"]).long()
        # target[int(self.normalized_dataset.iloc[index]["labels"])] = 1

        return img, target
        

    def _generate_tas(self, df, interval=range(6, 21)):

        for indicator in tqdm(self.indicators):
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

    def create_labels(self, df, col_name, window_size=11):
        """
        Data is labeled as per the logic in research paper
        Label code : BUY => 1, SELL => 0, HOLD => 2
        params :
            df => Dataframe with data
            col_name => name of column which should be used to determine strategy
        returns : numpy array with integer codes for labels with
                  size = total-(window_size)+1
        """

        row_counter = 0
        total_rows = len(df)
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
                    price = df.iloc[i][col_name]
                    if price < min_:
                        min_ = price
                        min_index = i
                    if price > max_:
                        max_ = price
                        max_index = i

                if max_index == window_middle:
                    labels[window_middle] = 0
                elif min_index == window_middle:
                    labels[window_middle] = 1
                else:
                    labels[window_middle] = 2

            row_counter = row_counter + 1
            pbar.update(1)

        pbar.close()
        return labels


# dataset = STIMDataset()
# dataloader = DataLoader(dataset, batch_size=8)
# hebe = next(iter(dataloader))
# hebe = 0
import numpy as np
from enum import Enum
import pandas as pd
from datetime import datetime

class Trend(Enum):
    Up = 1
    Down = -1
    Empty = 0

def find_next_lower(df, thresh, initial_point_idx = 0):
    try:
        if initial_point_idx == 0:
            return df[df["Low"] <= thresh].iloc[0].name
        else:
            sliced = df.loc[initial_point_idx:]
            return sliced[sliced["Low"] <= thresh].iloc[0].name
    except IndexError:
            return df.iloc[-1].name

def find_next_higher(df, thresh, initial_point_idx = 0):
    try:
        if initial_point_idx == 0:
            return df[df["High"] >= thresh].iloc[0].name
        else:
            sliced = df.loc[initial_point_idx:]
            return sliced[sliced["High"] >= thresh].iloc[0].name
    except IndexError:
            return df.iloc[-1].name      

def identify_init_trend(df, pct):
    init_price_high = df.iloc[0]["High"]
    init_price_low = df.iloc[0]["Low"]

    high_thresh = init_price_low * (1 + pct)
    low_thresh = init_price_high * (1 - pct)

    closest_new_low = find_next_lower(df, low_thresh)
    closest_new_high = find_next_higher(df, high_thresh)

    if closest_new_low < closest_new_high:
        df.loc[:closest_new_low, "trend"] = Trend.Down
        trend = Trend.Down
    else:
        df.loc[:closest_new_high, "trend"] = Trend.Up
        trend = Trend.Up

    return trend

def identify_next_trend(df, pct, trend):
    initial_point_idx = df[df["trend"] == trend].iloc[-1].name
    final_point_idx = df.iloc[-1].name

    init_price = df.loc[initial_point_idx, "High"] if trend == Trend.Up else df.loc[initial_point_idx, "Low"]

    high_thresh = init_price * (1 + pct)
    low_thresh = init_price * (1 - pct)

    if trend == Trend.Up:
        closest_new_low = find_next_lower(df, low_thresh, initial_point_idx)
        highers = df[df["High"] > init_price].loc[initial_point_idx:closest_new_low]
        if len(highers) > 0:
            if final_point_idx == closest_new_low:
                df.loc[initial_point_idx:, "trend"].iloc[10] = Trend.Up
            else:
                new_pivot = df.loc[initial_point_idx:closest_new_low, "High"].idxmax()
                df.loc[initial_point_idx:new_pivot, "trend"] = Trend.Up
            trend = Trend.Up
        else:
            df.loc[initial_point_idx:closest_new_low, "trend"] = Trend.Down
            trend = Trend.Down
    elif trend == Trend.Down:
        closest_new_high = find_next_higher(df, high_thresh, initial_point_idx)
        lowers = df[df["Low"] < init_price].loc[initial_point_idx:closest_new_high]
        if len(lowers) > 0:
            #
            if final_point_idx == closest_new_high:
                df.loc[initial_point_idx:, "trend"].iloc[10] = Trend.Down
            else:
                new_pivot = df.loc[initial_point_idx:closest_new_high, "Low"].idxmin()
                df.loc[initial_point_idx:new_pivot, "trend"] = Trend.Down
            trend = Trend.Down
        else:
            df.loc[initial_point_idx:closest_new_high, "trend"] = Trend.Up
            trend = Trend.Up

    return trend
        

def get_zigzag_pivots(df, pct=.09):
    df["trend"] = Trend.Empty
    trend = identify_init_trend(df, pct)
    while len(df[df["trend"] == Trend.Empty]) > 0:
        trend = identify_next_trend(df, pct, trend)

if __name__=="__main__":
    tickers = pd.read_csv("SPY.csv")
    tickers["Date"] = pd.to_datetime(tickers["Date"], format="%Y-%m-%d")
    tickers.set_index("Date", inplace=True)
    pivots = get_zigzag_pivots(tickers)
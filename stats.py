import numpy as np
import pandas as pd

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
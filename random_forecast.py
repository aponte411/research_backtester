from backtester import Strategy, Portfolio
import numpy as np
from pandas_datareader import DataReader
import pandas as pd
import scipy.stats as ss

import os
from typing import Any, Tuple
import click
os.chdir('/Users/davidaponte/TRADING/research-backtester/research_backtester')


class RandomForecastStrategy(Strategy):
    """
    Derives from Strategy to produce a set of signals that
    are randomly generated long/shorts.
    """

    def __init__(self, ticker: str, bars: pd.DataFrame):
        self.ticker = ticker
        self.bars = bars

    def generate_signals(self) -> pd.DataFrame:
        """
        Creates a pandas DataFrame of random signals
        from a normal distribution.
        """

        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = np.sign(ss.norm.rvs(size=len(signals)))
        signals['signal'][:5] = 0 # to offset upstream NaN errors

        return signals


class MarketOpenPortfolio(Portfolio):
    """
    Inherits Portfolio to create a system that purchases 100 units of
    a particular symbol upon a long/short signal, assuming the market
    open price of a bar.

    In addition, there are zero transaction costs and cash can be immediately
    borrowed for shorting (no margin posting or interest requirements).

    :symbol - A stock symbol which forms the basis of the portfolio.
    :bars - A DataFrame of bars for a symbol set.
    :signals - A pandas DataFrame of signals (1, 0, -1) for each symbol.
    :initial_capital - The amount in cash at the start of the portfolio.
    """

    def __init__(self,
                 ticker: str,
                 bars: pd.DataFrame,
                 signals: pd.DataFrame,
                 initial_capital: float = 100000.0):
        self.ticker = ticker
        self.bars = bars
        self.signals = signals
        self.initial_capital = initial_capital
        self.positions = self.generate_positions()

    def generate_positions(self) -> pd.DataFrame:
        """
        Creates a 'positions' DataFrame that simply longs or shorts
        100 of the particular symbol based on the forecast signals of
        {1, 0, -1} from the signals DataFrame.
        """

        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions[self.ticker] = 100*self.signals['signal']

        return positions

    def backtest_portfolio(self) -> pd.DataFrame:
        """
        Constructs a portfolio from the positions DataFrame by
        assuming the ability to trade at the precise market open price
        of each bar (an unrealistic assumption!).

        Calculates the total of cash and the holdings (market price of
        each position per bar), in order to generate an equity curve
        ('total') and a set of bar-based returns ('returns').

        Returns the portfolio object to be used elsewhere.
        """

        portfolio = self.positions*self.bars['Open']
        pos_diff = self.positions.diff()

        portfolio['holdings'] = (self.positions*self.bars['Open']).sum(axis=1)
        portfolio['cash'] = self.initial_capital - (pos_diff*self.bars['Open']).sum(axis=1).cumsum()

        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()

        return portfolio


def generate_stock_data(ticker: str,
                        source: str,
                        start: str,
                        end: str) -> Tuple[str, pd.DataFrame]:
    """Return stock ticker and bars"""

    bars = DataReader(ticker, source, start, end)

    return ticker, bars


@click.command()
@click.option('-tk', '--ticker', type=str, default='AAPL')
@click.option('-so', '--source', type=str, default='yahoo')
@click.option('-sd', '--start', type=str, default='2006-10-01')
@click.option('-ed', '--end', type=str, default='2018-10-01')
@click.option('-icap', '--initial-capital', type=float, default=100000.0)
def main(ticker: str,
         source: str,
         initial_capital: float,
         start: str,
         end: str) -> Any:

    ticker, bars = generate_stock_data(ticker=ticker,
                                       source=source,
                                       start=start,
                                       end=end)
    strategy = RandomForecastStrategy(ticker, bars)
    signals = strategy.generate_signals()
    portfolio = MarketOpenPortfolio(ticker, bars, signals, initial_capital=initial_capital)
    returns = portfolio.backtest_portfolio()

    print(f'Final returns for {ticker}..')
    print()
    print(returns.tail(10))


if __name__ == '__main__':
    main()

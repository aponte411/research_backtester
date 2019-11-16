from backtester import Strategy, Portfolio
import os

os.chdir('/Users/davidaponte/TRADING/research-backtester/research_backtester')
os.getcwd()

import numpy as np
import quandl
import pandas as pd
import scipy.stats as ss


class RandomForecastStrategy(Strategy):
    """
    Derives from Strategy to produce a set of signals that
    are randomly generated long/shorts. Clearly a nonsensical
    strategy, but perfectly acceptable for demonstrating the
    backtesting infrastructure!
    """
    def __init__(self, ticker, bars):
        self.ticker
        self.bars

    def generate_signals(self, ticker, bars):
        """
        Creates a pandas DataFrame of random signals
        from a Normal distribution.
        """
        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = np.sign(ss.norm.rvs(size=len(signals)))
        signals['signal'][:5] = 0 # to offset upstream NaN errors
        return signals

class MarketOpenPortfolio(Portfolio):

    def __init__(self, ticker, bars, signals, initial_capital=100000.0):
        self.ticker = ticker
        self.bars = bars
        self.signals = signals
        self.initial_capital = initial_capital
        self.positions = self.generate_positions()

    def generate_positions(self):
        """
        Creates a 'positions' DataFrame that simply longs or shorts
        100 of the particular symbol based on the forecast signals of
        {1, 0, -1} from the signals DataFrame.
        """
        positions = pd.DataFrame(index=self.signals.index).fillna(0.)
        positions[self.ticker] = 100*self.signals['signal']
        return positions

    def backtest_portfolio(self):
        """
        Constructs a portfolio from the positions DataFrame by
        assuming the ability to trade at the precise market open price
        of each bar (an unrealistic assumption!).

        Calculates the total of cash and the holdings (market price of
        each position per bar), in order to generate an equity curve
        ('total') and a set of bar-based returns ('returns').

        Returns the portfolio object to be used elsewhere."""

        portfolio = self.positions*self.bars['Open']
        pos_diff = self.positions.diff()

        portfolio['holdings'] = (self.positions*self.bars['Open']).sum(axis=1)
        portfolio['cash'] = self.initial_capital - (pos_diff*self.bars['Open']).sum(axis=1).cumsum()

        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_chg()
        return portfolio


def generate_stock_data(stock='WIKI/AAPL', collapse='daily'):
    ticker = stock.split('/')[1]
    bars = quandl.get(stock, collapse=collapse)
    return ticker, bars


def main():
    ticker, bars = generate_stock_data()
    strategy = RandomForecastStrategy(ticker, bars)
    signals = strategy.generate_signals()
    portfolio = MarketOpenPortfolio(ticker, bars, signals, initial_capital=100000.0)
    returns = portfolio.backtest_portfolio()
    print(returns.tail(10))

if __name__ == '__main__':
    main()

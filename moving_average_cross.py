from backtester import Strategy, Portfolio
import numpy as np
from pandas_datareader import DataReader
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import os
from typing import Any, Tuple
import click
import datetime
os.chdir('/Users/davidaponte/TRADING/research-backtester/research_backtester')


class MovingAverageCrossStrategy(Strategy):
    """
    Requires:
    symbol - A stock symbol on which to form a strategy on.
    bars - A DataFrame of bars for the above symbol.
    short_window - Lookback period for short moving average.
    long_window - Lookback period for long moving average.
    """

    def __init__(self,
                 ticker: str,
                 bars: pd.DataFrame,
                 short_window: int = 100,
                 long_window: int = 400) -> Any:
        self.ticker = ticker
        self.bars = bars
        self.short_window = short_window
        self.long_window = long_window


    def generate_signals(self) -> pd.DataFrame:
        """
        Returns the DataFrame of symbols containing the signals
        to go long, short or hold (1, -1 or 0).
        """

        signals = pd.DataFrame(index=self.bars.index)
        signals['signal'] = 0.0

        signals['short_mavg'] = self.bars['Close'].rolling(window=self.short_window).mean()
        signals['long_mavg'] = self.bars['Close'].rolling(window=self.long_window).mean()

        # Creates a 'signal' (invested or not invested) when the short moving average crosses the long
        # moving average, but only for the period greater than the shortest moving average window
        signals['signal'][self.short_window:] = np.where(signals['short_mavg'][self.short_window:] \
            > signals['long_mavg'][self.short_window:],
              1.0,
              0.0)
        signals['positions'] = signals['signal'].diff()

        return signals


class MarketOnClosePortfolio(Portfolio):
    """
    Encapsulates the notion of a portfolio of positions based
    on a set of signals as provided by a Strategy.

    Requires:
    symbol - A stock symbol which forms the basis of the portfolio.
    bars - A DataFrame of bars for a symbol set.
    signals - A pandas DataFrame of signals (1, 0, -1) for each symbol.
    initial_capital - The amount in cash at the start of the portfolio.
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
        positions[self.ticker] = 100*self.signals['signal'] # buy 100 shares

        return positions

    def backtest_portfolio(self) -> pd.DataFrame:
        """
        Constructs a portfolio from the positions DataFrame by
        assuming the ability to trade at the precise market close price
        of each bar (an unrealistic assumption!).

        Calculates the total of cash and the holdings (market price of
        each position per bar), in order to generate an equity curve
        ('total') and a set of bar-based returns ('returns').

        Returns the portfolio object to be used elsewhere.
        """

        portfolio = self.positions*self.bars['Close']
        pos_diff = self.positions.diff()

        portfolio['holdings'] = (self.positions*self.bars['Close']).sum(axis=1)
        portfolio['cash'] = self.initial_capital - (pos_diff*self.bars['Close']).sum(axis=1).cumsum()

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


def plot_equity_curves(bars: pd.DataFrame,
                       signals: pd.DataFrame,
                       returns: pd.DataFrame) -> Any:

    # fig, axes = fig.subplots(2, 1, figsize=(10,10))
    #
    # bars['Close'].plot(ax=axes[0], color='r', lw=2)
    # signals[['short_mavg', 'long_mavg']].plot(ax=axes[1], lw=2)
    # Plot two charts to assess trades and equity curve
    fig = plt.figure(figsize=(8,7))
    fig.patch.set_facecolor('white')     # Set the outer colour to white
    ax1 = fig.add_subplot(211,  ylabel='Price in $')

    # Plot the AAPL closing price overlaid with the moving averages
    bars['Close'].plot(ax=ax1, color='r', lw=2.)
    signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

    # Plot the "buy" trades against AAPL
    ax1.plot(signals.loc[signals.positions == 1.0].index,
             signals.short_mavg[signals.positions == 1.0],
             '^', markersize=10, color='m')

    # Plot the "sell" trades against AAPL
    ax1.plot(signals.loc[signals.positions == -1.0].index,
             signals.short_mavg[signals.positions == -1.0],
             'v', markersize=10, color='k')

    # Plot the equity curve in dollars
    ax2 = fig.add_subplot(212, ylabel='Portfolio value in $')
    returns['total'].plot(ax=ax2, lw=2.)

    # Plot the "buy" and "sell" trades against the equity curve
    ax2.plot(returns.loc[signals.positions == 1.0].index,
             returns.total[signals.positions == 1.0],
             '^', markersize=10, color='m')
    ax2.plot(returns.loc[signals.positions == -1.0].index,
             returns.total[signals.positions == -1.0],
             'v', markersize=10, color='k')

    # Plot the figure
    fig.show()

# ticker, bars = generate_stock_data(ticker='AAPL',
#                                    source='yahoo',
#                                    start='1990-01-01',
#                                    end='2002-01-01')
# macs = MovingAverageCrossStrategy(ticker,
#                                   bars,
#                                   short_window=100,
#                                   long_window=400)
# signals = macs.generate_signals()
#
# portfolio = MarketOnClosePortfolio(ticker,
#                                    bars,
#                                    signals,
#                                    initial_capital=100000.0)
# returns = portfolio.backtest_portfolio()
# returns
# plot_equity_curves(bars, signals)

@click.command()
@click.option('-tk', '--ticker', type=str, default='AAPL')
@click.option('-so', '--source', type=str, default='yahoo')
@click.option('-sd', '--start', type=str, default='1990-01-01')
@click.option('-ed', '--end', type=str, default='2002-01-01')
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
    macs = MovingAverageCrossStrategy(ticker,
                                      bars,
                                      short_window=100,
                                      long_window=400)
    signals = macs.generate_signals()

    portfolio = MarketOnClosePortfolio(ticker,
                                       bars,
                                       signals,
                                       initial_capital=initial_capital)
    returns = portfolio.backtest_portfolio()
    plot_equity_curves(bars, signals, returns)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

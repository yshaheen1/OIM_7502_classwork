import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sb
import yfinance as yf

sb.set_theme()

DEFAULT_START = (dt.date.today() - dt.timedelta(365)).isoformat()
DEFAULT_END = dt.date.today().isoformat()

class Stock:
    def __init__(self, symbol, start=DEFAULT_START, end=DEFAULT_END):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.data = self.get_data()

    def get_data(self):
        """Fetch historical data and add returns."""
        df = yf.download(self.symbol, start=self.start, end=self.end, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {self.symbol}.")
        df.index = pd.to_datetime(df.index)
        return self.calc_returns(df)

    def calc_returns(self, df):
        """Add change and instantaneous (log) return."""
        df["change"] = df["Close"].diff()
        df["instant_return"] = np.log(df["Close"]).diff().round(4)
        return df

    def plot_return_dist(self, bins=50):
        r = self.data["instant_return"].dropna()
        plt.hist(r, bins=bins, edgecolor="white")
        plt.axvline(r.mean(), linestyle="--", color="black", label=f"Mean: {r.mean():.4f}")
        plt.title(f"{self.symbol} — Daily Instantaneous Return Distribution")
        plt.xlabel("Log return")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, linestyle=":")
        plt.show()

    def plot_performance(self):
        perf = self.data["Close"] / self.data["Close"].iloc[0] - 1
        plt.plot(perf.index, perf.values, label=self.symbol)
        plt.axhline(0, color="gray", linestyle="--")
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.title(f"{self.symbol} — Performance Since Start (% gain/loss)")
        plt.xlabel("Date")
        plt.ylabel("Return (%)")
        plt.legend()
        plt.grid(True, linestyle=":")
        plt.show()


def main():
    stock = Stock("MSFT")
    print(stock.data.tail())
    stock.plot_performance()
    stock.plot_return_dist()


if __name__ == "__main__":
    main()

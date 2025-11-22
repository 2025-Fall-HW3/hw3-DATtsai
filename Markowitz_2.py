"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=20, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """

        def normalize(x):
            mx, mn = np.nanmax(x), np.nanmin(x)
            if mx != mn:
                return (x - mn) / (mx - mn)
            else:
                return np.zeros(len(x))

        R_n = self.returns[assets]
        w = np.ones(len(assets)) / len(assets)

        for i in range(self.lookback, len(self.price)):
            
            R_window = R_n.iloc[i - self.lookback : i]
            R_mean = R_window.mean().values
            R_mean_positive = np.where(R_mean > 0, R_mean, 0)
            R_std = np.where(R_window.std().values < 1e-9, 1e-9, R_window.std().values)
            R_mean_positive_std = np.where(R_mean > 0, R_std, 1e-9)
            inverse_R_std = 1 / R_std
            inverse_R_mean_positive_std = np.where(R_mean_positive_std > 0, 1 / R_mean_positive_std, 0)
            R_residual = R_mean - R_mean.mean()
            R_residual_std = R_residual.std()
            penalty = R_std / R_std.mean()
            between_std = R_std - R_residual_std

            # 定義score
            # score = R_mean / R_std # sharp ratio like
            # score = score / penalty
            # score = 1 / R_std
            # score = inverse_R_std / inverse_R_std.sum()
            # score = R_mean * (1/penalty) / R_residual_std
            # score = 1 / (R_std - R_residual_std)
            # score = R_residual / penalty
            # score = np.where(score > 0, score, 0) * (-between_std)
            score = R_mean_positive / R_mean_positive_std
            # score = 0.4 * normalize(R_mean_positive) + 0.6 * normalize(inverse_R_mean_positive_std) # + 0.05 * normalize(R_residual) + 0.05 * normalize(-between_std)
            
            score = normalize(score)
            w = score
            
            if w.sum() == 0:
                # w = np.ones(len(assets)) / len(assets)
                w = np.zeros(len(assets))
            else:
                w = w / w.sum() # 除以總和做正規化
                # w = np.minimum(w, 0.3) # 最大權重限制
                # w = w / w.sum()

            self.portfolio_weights.loc[self.price.index[i], assets] = w
        
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()
        
        # print('weight matrix: boosting weight')
        # print(self.portfolio_weights)
        # print('returns: ')
        # print(self.portfolio_returns)
        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)

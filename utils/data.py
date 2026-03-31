"""
data.py — Fetch and process fund/ETF data using yfinance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


PERIOD_MAP = {
    "1 month": "1mo",
    "3 months": "3mo",
    "6 months": "6mo",
    "1 year": "1y",
    "3 years": "3y",
    "5 years": "5y",
}


def fetch_fund_info(ticker: str) -> dict:
    """
    Fetch basic fund metadata from Yahoo Finance.
    Returns a dict with name, category, expense ratio, etc.
    """
    try:
        fund = yf.Ticker(ticker)
        info = fund.info

        return {
            "ticker": ticker.upper(),
            "name": info.get("longName") or info.get("shortName", "N/A"),
            "category": info.get("category", "N/A"),
            "asset_class": info.get("quoteType", "N/A"),
            "currency": info.get("currency", "N/A"),
            "exchange": info.get("exchange", "N/A"),
            "expense_ratio": info.get("annualReportExpenseRatio") or info.get("expenseRatio", None),
            "aum": info.get("totalAssets", None),
            "pe_ratio": info.get("trailingPE", None),
            "dividend_yield": info.get("dividendYield", None),
            "ytd_return": info.get("ytdReturn", None),
            "three_year_return": info.get("threeYearAverageReturn", None),
            "five_year_return": info.get("fiveYearAverageReturn", None),
            "beta": info.get("beta3Year") or info.get("beta", None),
            "description": info.get("longBusinessSummary", ""),
            "sector_weightings": info.get("sectorWeightings", []),
            "holdings": info.get("holdings", []),
        }
    except Exception as e:
        return {"ticker": ticker.upper(), "error": str(e)}


def fetch_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch historical closing prices for a ticker.
    Returns a DataFrame with Date and Close columns.
    """
    try:
        fund = yf.Ticker(ticker)
        hist = fund.history(period=period)
        if hist.empty:
            return pd.DataFrame()
        hist = hist[["Close"]].rename(columns={"Close": ticker.upper()})
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        return hist
    except Exception:
        return pd.DataFrame()


def compute_metrics(ticker: str, period: str = "1y") -> dict:
    """
    Compute quantitative metrics: total return, annualized vol, Sharpe ratio, max drawdown.
    """
    hist = fetch_price_history(ticker, period)
    if hist.empty:
        return {}

    prices = hist.iloc[:, 0].dropna()
    if len(prices) < 2:
        return {}

    daily_returns = prices.pct_change().dropna()

    # Total return
    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1

    # Annualized volatility (252 trading days)
    ann_vol = daily_returns.std() * np.sqrt(252)

    # Sharpe ratio (assuming risk-free rate = 4%)
    rf_daily = 0.04 / 252
    excess_returns = daily_returns - rf_daily
    sharpe = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else None

    # Max drawdown
    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        "total_return": round(total_return * 100, 2),
        "annualized_volatility": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 2) if sharpe else None,
        "max_drawdown": round(max_drawdown * 100, 2),
        "current_price": round(prices.iloc[-1], 2),
        "data_points": len(prices),
    }


def build_fund_summary_dict(ticker: str, period: str = "1y") -> dict:
    """
    Aggregate all fund data into a single dict ready for display or LLM input.
    """
    info = fetch_fund_info(ticker)
    if "error" in info:
        return info

    metrics = compute_metrics(ticker, period)
    info.update(metrics)
    return info


def format_aum(value) -> str:
    """Format AUM in human-readable form."""
    if value is None:
        return "N/A"
    if value >= 1e12:
        return f"${value / 1e12:.1f}T"
    if value >= 1e9:
        return f"${value / 1e9:.1f}B"
    if value >= 1e6:
        return f"${value / 1e6:.1f}M"
    return f"${value:,.0f}"


def format_pct(value) -> str:
    """Format a ratio as percentage string."""
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%" if abs(value) < 1 else f"{value:.2f}%"

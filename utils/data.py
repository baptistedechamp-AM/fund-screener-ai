"""
data.py — Fetch and process fund/ETF data using yfinance + FMP fallback.
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta


PERIOD_MAP = {
    "YTD": "ytd",
    "1 month": "1mo",
    "3 months": "3mo",
    "6 months": "6mo",
    "1 year (rolling)": "1y",
    "3 years (rolling)": "3y",
    "5 years (rolling)": "5y",
}

PRESET_FUNDS = {
    "US Core": ["SPY", "QQQ", "AGG", "VTI", "BND"],
    "UCITS (EU-listed)": ["IWDA.L", "VWCE.DE", "CSPX.L", "EUNL.DE"],
    "European Equity": ["EXW1.DE", "VEUR.L", "CEUL.L", "IEER.L"],
    "Global Diversified": ["SPY", "IWDA.L", "AGG", "GLD", "VWO"],
}

BENCHMARK_MAP = {
    "large blend": "SPY",
    "large growth": "QQQ",
    "large value": "VTV",
    "mid-cap": "IJH",
    "small blend": "IWM",
    "small growth": "IWO",
    "technology": "XLK",
    "health": "XLV",
    "energy": "XLE",
    "financial": "XLF",
    "world": "URTH",
    "global": "URTH",
    "international": "VXUS",
    "europe": "VGK",
    "emerging": "VWO",
    "pacific": "VPL",
    "bond": "AGG",
    "fixed income": "AGG",
    "aggregate bond": "AGG",
    "corporate bond": "LQD",
    "treasury": "TLT",
    "high yield": "HYG",
    "short-term bond": "SHV",
    "commodity": "DBC",
    "gold": "GLD",
    "real estate": "VNQ",
}


def guess_benchmark(fund_data: dict) -> str:
    category = (fund_data.get("category") or "").lower()
    description = (fund_data.get("description") or "").lower()
    name = (fund_data.get("name") or "").lower()
    search_text = f"{category} {description} {name}"
    for keyword, bench_ticker in BENCHMARK_MAP.items():
        if keyword in search_text:
            return bench_ticker
    return "SPY"


# ─── FMP fallback ─────────────────────────────────────────────────────────────

def fetch_fmp_etf_info(ticker: str) -> dict:
    """Fetch ETF metadata from Financial Modeling Prep as fallback."""
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return {}
    
    # Strip exchange suffix for FMP (e.g. IWDA.L -> IWDA)
    clean_ticker = ticker.split(".")[0].upper()
    
    try:
        url = f"https://financialmodelingprep.com/api/v3/etf-info?symbol={clean_ticker}&apikey={api_key}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data and isinstance(data, list) and len(data) > 0:
            info = data[0]
            return {
                "fmp_expense_ratio": info.get("expenseRatio"),
                "fmp_aum": info.get("aum") or info.get("totalAssets"),
                "fmp_nav": info.get("navPrice"),
                "fmp_holdings_count": info.get("holdingsCount"),
            }
    except Exception:
        pass
    return {}


# ─── Yahoo Finance data ──────────────────────────────────────────────────────

def fetch_fund_info(ticker: str) -> dict:
    """Fetch fund metadata from Yahoo Finance, with FMP fallback."""
    try:
        fund = yf.Ticker(ticker)
        info = fund.info

        yf_data = {
            "ticker": ticker.upper(),
            "name": info.get("longName") or info.get("shortName", "N/A"),
            "category": info.get("category", "N/A"),
            "asset_class": info.get("quoteType", "N/A"),
            "currency": info.get("currency", "N/A"),
            "exchange": info.get("exchange", "N/A"),
            "expense_ratio": info.get("annualReportExpenseRatio") or info.get("expenseRatio", None),
            "aum": info.get("totalAssets", None),
            "pe_ratio": info.get("trailingPE", None),
            "dividend_yield": info.get("dividendYield", None) if (info.get("dividendYield") or 0) < 0.30 else None,
            "ytd_return": info.get("ytdReturn", None),
            "three_year_return": info.get("threeYearAverageReturn", None),
            "five_year_return": info.get("fiveYearAverageReturn", None),
            "beta": info.get("beta3Year") or info.get("beta", None),
            "description": info.get("longBusinessSummary", ""),
            "sector_weightings": info.get("sectorWeightings", []),
            "holdings": info.get("holdings", []),
        }

        # FMP fallback for missing metadata
        if not yf_data["expense_ratio"] or not yf_data["aum"]:
            fmp = fetch_fmp_etf_info(ticker)
            if not yf_data["expense_ratio"] and fmp.get("fmp_expense_ratio"):
                yf_data["expense_ratio"] = fmp["fmp_expense_ratio"]
            if not yf_data["aum"] and fmp.get("fmp_aum"):
                yf_data["aum"] = fmp["fmp_aum"]
            if fmp.get("fmp_holdings_count"):
                yf_data["holdings_count"] = fmp["fmp_holdings_count"]

        return yf_data
    except Exception as e:
        return {"ticker": ticker.upper(), "error": str(e)}


def fetch_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch historical closing prices."""
    try:
        fund = yf.Ticker(ticker)
        if period == "ytd":
            hist = fund.history(period="ytd")
        else:
            hist = fund.history(period=period)
        if hist.empty:
            return pd.DataFrame()
        hist = hist[["Close"]].rename(columns={"Close": ticker.upper()})
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        return hist
    except Exception:
        return pd.DataFrame()


def compute_metrics(ticker: str, period: str = "1y", fund_info: dict = None) -> dict:
    """Compute quantitative metrics."""
    hist = fetch_price_history(ticker, period)
    if hist.empty:
        return {}

    prices = hist.iloc[:, 0].dropna()
    if len(prices) < 2:
        return {}

    daily_returns = prices.pct_change().dropna()
    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    ann_vol = daily_returns.std() * np.sqrt(252)

    rf_daily = 0.04 / 252
    excess_returns = daily_returns - rf_daily
    sharpe = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else None

    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (excess_returns.mean() * 252) / downside_std if downside_std > 0 else None

    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    tracking_error = None
    benchmark_ticker = None
    if fund_info:
        benchmark_ticker = guess_benchmark(fund_info)
        if benchmark_ticker.upper() != ticker.upper():
            try:
                bench_hist = fetch_price_history(benchmark_ticker, period)
                if not bench_hist.empty:
                    bench_returns = bench_hist.iloc[:, 0].pct_change().dropna()
                    common_idx = daily_returns.index.intersection(bench_returns.index)
                    if len(common_idx) > 20:
                        diff = daily_returns.loc[common_idx] - bench_returns.loc[common_idx]
                        tracking_error = round(diff.std() * np.sqrt(252) * 100, 2)
            except Exception:
                pass

    return {
        "total_return": round(total_return * 100, 2),
        "annualized_volatility": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 2) if sharpe else None,
        "sortino_ratio": round(sortino, 2) if sortino else None,
        "max_drawdown": round(max_drawdown * 100, 2),
        "tracking_error": tracking_error,
        "benchmark_used": benchmark_ticker,
        "current_price": round(prices.iloc[-1], 2),
        "data_points": len(prices),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def detect_significant_moves(ticker: str, period: str = "1y", threshold: float = 2.0) -> list:
    """Detect days with abnormally large moves."""
    hist = fetch_price_history(ticker, period)
    if hist.empty:
        return []

    prices = hist.iloc[:, 0].dropna()
    daily_returns = prices.pct_change().dropna()
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()

    if std_ret == 0:
        return []

    events = []
    for date, ret in daily_returns.items():
        z_score = abs((ret - mean_ret) / std_ret)
        if z_score >= threshold:
            events.append({
                "date": date,
                "return_pct": round(ret * 100, 2),
                "z_score": round(z_score, 1),
                "price": round(prices.loc[date], 2),
            })

    events.sort(key=lambda x: abs(x["return_pct"]), reverse=True)
    return events[:5]


def build_fund_summary_dict(ticker: str, period: str = "1y") -> dict:
    info = fetch_fund_info(ticker)
    if "error" in info:
        return info
    metrics = compute_metrics(ticker, period, fund_info=info)
    info.update(metrics)
    return info


def format_aum(value) -> str:
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
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%" if abs(value) < 1 else f"{value:.2f}%"
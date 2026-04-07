"""
data.py — Fetch and process fund/ETF data using yfinance + FMP fallback.
Includes hardcoded metadata for European UCITS ETFs (from justETF)
where yfinance and FMP fail to provide expense ratios, categories, etc.
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from urllib.parse import quote_plus


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
    "European Equity": ["EXW1.DE", "VEUR.L", "MEUD.PA", "IMAE.AS"],
    "Global Diversified": ["SPY", "IWDA.L", "AGG", "GLD", "VWO"],
}

# ─── Hardcoded European ETF metadata (source: justETF, April 2026) ────────────
# yfinance and FMP free tier often fail for EU-listed ETFs.
# This dict provides reliable fallback data for popular UCITS ETFs.

EU_ETF_METADATA = {
    # ── UCITS preset ──
    "IWDA.L": {
        "category": "Global Equity – MSCI World",
        "expense_ratio": 0.0020,
        "aum": 123_000_000_000,
        "distribution": "Acc",
        "index": "MSCI World",
        "provider": "iShares (BlackRock)",
    },
    "VWCE.DE": {
        "category": "Global Equity – FTSE All-World",
        "expense_ratio": 0.0019,
        "aum": 31_500_000_000,
        "distribution": "Acc",
        "index": "FTSE All-World",
        "provider": "Vanguard",
    },
    "CSPX.L": {
        "category": "US Equity – S&P 500",
        "expense_ratio": 0.0007,
        "aum": 107_000_000_000,
        "distribution": "Acc",
        "index": "S&P 500",
        "provider": "iShares (BlackRock)",
    },
    "EUNL.DE": {
        "category": "Global Equity – MSCI World",
        "expense_ratio": 0.0020,
        "aum": 123_000_000_000,
        "distribution": "Acc",
        "index": "MSCI World",
        "provider": "iShares (BlackRock)",
    },
    # ── European Equity preset ──
    "EXW1.DE": {
        "category": "Eurozone Equity – EURO STOXX 50",
        "expense_ratio": 0.0009,
        "aum": 9_600_000_000,
        "dividend_yield": 0.0252,
        "distribution": "Dist",
        "index": "EURO STOXX 50",
        "provider": "iShares (BlackRock)",
    },
    "VEUR.L": {
        "category": "European Equity – FTSE Developed Europe",
        "expense_ratio": 0.0010,
        "aum": 4_200_000_000,
        "dividend_yield": 0.0310,
        "distribution": "Dist",
        "index": "FTSE Developed Europe",
        "provider": "Vanguard",
    },
    "MEUD.PA": {
        "category": "European Equity – STOXX Europe 600",
        "expense_ratio": 0.0020,
        "aum": 8_500_000_000,
        "distribution": "Acc",
        "index": "STOXX Europe 600",
        "provider": "Amundi",
    },
    "IMAE.AS": {
        "category": "European Equity – MSCI Europe",
        "expense_ratio": 0.0012,
        "aum": 15_500_000_000,
        "distribution": "Acc",
        "index": "MSCI Europe",
        "provider": "iShares (BlackRock)",
    },
    # ── Other common UCITS ETFs ──
    "VWRL.L": {
        "category": "Global Equity – FTSE All-World",
        "expense_ratio": 0.0019,
        "dividend_yield": 0.0180,
        "distribution": "Dist",
        "index": "FTSE All-World",
        "provider": "Vanguard",
    },
    "SWDA.L": {
        "category": "Global Equity – MSCI World",
        "expense_ratio": 0.0020,
        "distribution": "Acc",
        "index": "MSCI World",
        "provider": "iShares (BlackRock)",
    },
    "VUSA.L": {
        "category": "US Equity – S&P 500",
        "expense_ratio": 0.0007,
        "dividend_yield": 0.0130,
        "distribution": "Dist",
        "index": "S&P 500",
        "provider": "Vanguard",
    },
    "EMIM.L": {
        "category": "Emerging Markets Equity – MSCI EM IMI",
        "expense_ratio": 0.0018,
        "distribution": "Acc",
        "index": "MSCI Emerging Markets IMI",
        "provider": "iShares (BlackRock)",
    },
    # ── Common US ETFs (fallback when yfinance returns None) ──
    "SPY": {
        "category": "US Equity – S&P 500",
        "expense_ratio": 0.0009,
        "dividend_yield": 0.0125,
        "distribution": "Dist",
        "index": "S&P 500",
        "provider": "State Street (SPDR)",
    },
    "QQQ": {
        "category": "US Equity – Nasdaq 100",
        "expense_ratio": 0.0020,
        "dividend_yield": 0.0055,
        "distribution": "Dist",
        "index": "Nasdaq 100",
        "provider": "Invesco",
    },
    "AGG": {
        "category": "US Bonds – Bloomberg US Aggregate",
        "expense_ratio": 0.0003,
        "dividend_yield": 0.0340,
        "distribution": "Dist",
        "index": "Bloomberg US Aggregate Bond",
        "provider": "iShares (BlackRock)",
    },
    "VTI": {
        "category": "US Equity – Total Stock Market",
        "expense_ratio": 0.0003,
        "dividend_yield": 0.0130,
        "distribution": "Dist",
        "index": "CRSP US Total Market",
        "provider": "Vanguard",
    },
    "BND": {
        "category": "US Bonds – Bloomberg US Aggregate",
        "expense_ratio": 0.0003,
        "dividend_yield": 0.0330,
        "distribution": "Dist",
        "index": "Bloomberg US Aggregate Float Adjusted",
        "provider": "Vanguard",
    },
    "GLD": {
        "category": "Commodity – Gold",
        "expense_ratio": 0.0040,
        "distribution": "None",
        "index": "Gold Spot Price",
        "provider": "State Street (SPDR)",
    },
    "VWO": {
        "category": "Emerging Markets Equity – FTSE EM",
        "expense_ratio": 0.0008,
        "dividend_yield": 0.0310,
        "distribution": "Dist",
        "index": "FTSE Emerging Markets All Cap China A Inclusion",
        "provider": "Vanguard",
    },
}


BENCHMARK_MAP = {
    # ── Bonds (check first — "bond" appears in many fund names/descriptions) ──
    "aggregate bond": "AGG",
    "bond": "AGG",
    "fixed income": "AGG",
    "corporate bond": "LQD",
    "treasury": "TLT",
    "high yield": "HYG",
    "short-term bond": "SHV",
    # ── Specific indices (check before broad keywords) ──
    "msci world": "URTH",
    "ftse all-world": "URTH",
    "s&p 500": "SPY",
    "euro stoxx": "VGK",
    "stoxx europe": "VGK",
    "ftse developed europe": "VGK",
    "msci europe": "VGK",
    # ── Regional / broad ──
    "europe": "VGK",
    "emerging": "SPY",
    "pacific": "VPL",
    "world": "URTH",
    "global": "URTH",
    # ── Style boxes ──
    "large blend": "SPY",
    "large growth": "SPY",
    "large value": "SPY",
    "mid-cap": "IJH",
    "small blend": "IWM",
    "small growth": "IWO",
    # ── Sectors ──
    "technology": "XLK",
    "health": "XLV",
    "energy": "XLE",
    "financial": "XLF",
    "real estate": "VNQ",
    "commodity": "SPY",
    "gold": "SPY",
    # ── International (LAST — avoid matching "international" in bond fund descriptions) ──
    "international": "VXUS",
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

def _fmp_api_key() -> str:
    return os.getenv("FMP_API_KEY", "")


def fetch_fmp_etf_info(ticker: str) -> dict:
    api_key = _fmp_api_key()
    if not api_key:
        return {}
    clean_ticker = ticker.split(".")[0].upper()
    tickers_to_try = [clean_ticker, ticker.upper()]
    for t in tickers_to_try:
        try:
            url = f"https://financialmodelingprep.com/api/v3/etf-info?symbol={t}&apikey={api_key}"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if data and isinstance(data, list) and len(data) > 0:
                info = data[0]
                result = {}
                if info.get("expenseRatio"):
                    result["fmp_expense_ratio"] = info["expenseRatio"]
                if info.get("aum") or info.get("totalAssets"):
                    result["fmp_aum"] = info.get("aum") or info.get("totalAssets")
                if info.get("navPrice"):
                    result["fmp_nav"] = info["navPrice"]
                if info.get("holdingsCount"):
                    result["fmp_holdings_count"] = info["holdingsCount"]
                if result:
                    return result
        except Exception:
            pass
    return {}


def fetch_fmp_profile(ticker: str) -> dict:
    api_key = _fmp_api_key()
    if not api_key:
        return {}
    clean_ticker = ticker.split(".")[0].upper()
    tickers_to_try = [clean_ticker, ticker.upper()]
    for t in tickers_to_try:
        try:
            url = f"https://financialmodelingprep.com/api/v3/profile/{t}?apikey={api_key}"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if data and isinstance(data, list) and len(data) > 0:
                info = data[0]
                result = {}
                if info.get("companyName"):
                    result["fmp_name"] = info["companyName"]
                if info.get("description"):
                    result["fmp_description"] = info["description"]
                if info.get("sector"):
                    result["fmp_sector"] = info["sector"]
                if info.get("beta"):
                    result["fmp_beta"] = info["beta"]
                if result:
                    return result
        except Exception:
            pass
    return {}


# ─── Google News links for market events ──────────────────────────────────────

def build_news_search_url(ticker: str, event_date, return_pct: float = None) -> str:
    """
    Build a Google News search URL with precise date filtering.
    Uses Google's tbs parameter to restrict results to a 3-day window.
    """
    if hasattr(event_date, "strftime"):
        dt = event_date
    else:
        dt = pd.Timestamp(event_date)

    clean_name = ticker.split(".")[0]

    eu_meta = EU_ETF_METADATA.get(ticker.upper(), {})
    index_name = eu_meta.get("index")

    etf_names = {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "AGG": "US bond market",
        "VTI": "US stock market",
        "BND": "US bond market",
        "GLD": "gold price",
        "VWO": "emerging markets stocks",
    }

    if index_name:
        search_name = index_name
    else:
        search_name = etf_names.get(clean_name, f"{ticker} ETF")

    # Build search query
    if return_pct is not None:
        direction = "drop fall decline" if return_pct < 0 else "rally surge gain"
        query = f"{search_name} stock market {direction}"
    else:
        query = f"{search_name} stock market news"

    # Google date filter: cd_min and cd_max in MM/DD/YYYY format
    date_from = (dt - timedelta(days=1)).strftime("%m/%d/%Y")
    date_to = (dt + timedelta(days=1)).strftime("%m/%d/%Y")

    encoded_query = quote_plus(query)
    date_filter = quote_plus(f"cdr:1,cd_min:{date_from},cd_max:{date_to}")

    return f"https://www.google.com/search?q={encoded_query}&tbm=nws&tbs={date_filter}"


# ─── Data cleaning ────────────────────────────────────────────────────────────

def _clean_price_data(prices: pd.Series, max_daily_change: float = 0.15) -> pd.Series:
    if len(prices) < 3:
        return prices

    cleaned = prices.copy()
    daily_returns = prices.pct_change()

    for i in range(1, len(prices) - 1):
        ret_today = daily_returns.iloc[i]
        ret_tomorrow = daily_returns.iloc[i + 1]
        if abs(ret_today) > max_daily_change and abs(ret_tomorrow) > max_daily_change:
            if (ret_today > 0 and ret_tomorrow < 0) or (ret_today < 0 and ret_tomorrow > 0):
                cleaned.iloc[i] = np.nan

    if len(daily_returns) > 2:
        last_ret = daily_returns.iloc[-1]
        if abs(last_ret) > max_daily_change:
            two_day_ret = (prices.iloc[-1] / prices.iloc[-3]) - 1
            if abs(two_day_ret) < max_daily_change * 0.5:
                cleaned.iloc[-1] = np.nan

    if cleaned.isna().any():
        cleaned = cleaned.interpolate(method="linear")

    return cleaned


# ─── Yahoo Finance data ──────────────────────────────────────────────────────

def fetch_fund_info(ticker: str) -> dict:
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
            "expense_ratio": (
                info.get("annualReportExpenseRatio")
                or info.get("expenseRatio")
                or None
            ),
            "aum": info.get("totalAssets", None),
            "pe_ratio": info.get("trailingPE", None),
            "dividend_yield": info.get("dividendYield", None) if (info.get("dividendYield") or 0) < 0.30 else None,
            "ytd_return": info.get("ytdReturn", None),
            "three_year_return": info.get("threeYearAverageReturn", None),
            "five_year_return": info.get("fiveYearAverageReturn", None),
            "beta": None,  # Computed in compute_metrics vs the right benchmark
            "description": info.get("longBusinessSummary", ""),
            "sector_weightings": info.get("sectorWeightings", []),
            "holdings": info.get("holdings", []),
        }

        # FMP fallback
        if not yf_data["expense_ratio"] or not yf_data["aum"]:
            fmp = fetch_fmp_etf_info(ticker)
            if not yf_data["expense_ratio"] and fmp.get("fmp_expense_ratio"):
                yf_data["expense_ratio"] = fmp["fmp_expense_ratio"]
            if not yf_data["aum"] and fmp.get("fmp_aum"):
                yf_data["aum"] = fmp["fmp_aum"]
            if fmp.get("fmp_holdings_count"):
                yf_data["holdings_count"] = fmp["fmp_holdings_count"]

        # FMP profile fallback for name and description
        needs_profile = (
            yf_data["name"] == "N/A"
            or not yf_data["description"]
        )
        if needs_profile:
            profile = fetch_fmp_profile(ticker)
            if yf_data["name"] == "N/A" and profile.get("fmp_name"):
                yf_data["name"] = profile["fmp_name"]
            if not yf_data["description"] and profile.get("fmp_description"):
                yf_data["description"] = profile["fmp_description"]

        # ── Hardcoded EU ETF fallback (justETF data) ──
        eu_meta = EU_ETF_METADATA.get(ticker.upper(), {})
        if eu_meta:
            if not yf_data["expense_ratio"] and eu_meta.get("expense_ratio"):
                yf_data["expense_ratio"] = eu_meta["expense_ratio"]
            if (yf_data["category"] == "N/A" or not yf_data["category"]) and eu_meta.get("category"):
                yf_data["category"] = eu_meta["category"]
            if not yf_data["dividend_yield"] and eu_meta.get("dividend_yield"):
                yf_data["dividend_yield"] = eu_meta["dividend_yield"]
            if eu_meta.get("distribution"):
                yf_data["distribution"] = eu_meta["distribution"]
            if eu_meta.get("index"):
                yf_data["index_tracked"] = eu_meta["index"]
            if eu_meta.get("provider"):
                yf_data["provider"] = eu_meta["provider"]
            if not yf_data["aum"] and eu_meta.get("aum"):
                yf_data["aum"] = eu_meta["aum"]

        return yf_data
    except Exception as e:
        return {"ticker": ticker.upper(), "error": str(e)}


def fetch_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
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

        col = ticker.upper()
        hist[col] = _clean_price_data(hist[col])

        return hist
    except Exception:
        return pd.DataFrame()


def compute_metrics(ticker: str, period: str = "1y", fund_info: dict = None) -> dict:
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
    computed_beta = None
    if fund_info:
        benchmark_ticker = guess_benchmark(fund_info)

        # If benchmark == ticker itself, use SPY as universal equity benchmark
        if benchmark_ticker.upper() == ticker.upper():
            # For SPY itself: beta = 1.00 by definition
            if ticker.upper() == "SPY":
                computed_beta = 1.00
                benchmark_ticker = "SPY"
            else:
                # For AGG, VWO, etc: compute beta vs SPY (broad market)
                benchmark_ticker = "SPY"

        # Check if we're using a real matched benchmark or a default fallback
        is_default_spy = (benchmark_ticker == "SPY"
                          and "s&p 500" not in (fund_info.get("category") or "").lower()
                          and "s&p 500" not in (fund_info.get("index_tracked") or "").lower()
                          and ticker.upper() != "SPY")

        # Compute beta + TE vs the benchmark
        if computed_beta is None and benchmark_ticker.upper() != ticker.upper():
            try:
                bench_hist = fetch_price_history(benchmark_ticker, period)
                if not bench_hist.empty:
                    bench_prices = bench_hist.iloc[:, 0].dropna()
                    
                    # Use WEEKLY returns for beta calculation
                    # This avoids EU/US calendar mismatch (different trading days)
                    # and gives more stable beta estimates
                    fund_weekly = prices.resample("W-FRI").last().dropna()
                    bench_weekly = bench_prices.resample("W-FRI").last().dropna()
                    common_weeks = fund_weekly.index.intersection(bench_weekly.index)
                    
                    if len(common_weeks) > 10:
                        fw_rets = fund_weekly.loc[common_weeks].pct_change().dropna()
                        bw_rets = bench_weekly.loc[common_weeks].pct_change().dropna()
                        common_ret_idx = fw_rets.index.intersection(bw_rets.index)
                        
                        if len(common_ret_idx) > 8:
                            fw = fw_rets.loc[common_ret_idx]
                            bw = bw_rets.loc[common_ret_idx]
                            
                            cov = np.cov(fw, bw)
                            if cov[1, 1] > 0:
                                computed_beta = round(cov[0, 1] / cov[1, 1], 2)
                    
                    # Tracking error — use daily returns on common days
                    if not is_default_spy:
                        bench_returns = bench_prices.pct_change().dropna()
                        common_idx = daily_returns.index.intersection(bench_returns.index)
                        if len(common_idx) > 20:
                            diff = daily_returns.loc[common_idx] - bench_returns.loc[common_idx]
                            te = round(diff.std() * np.sqrt(252) * 100, 2)
                            if te < 15:
                                tracking_error = te
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
        "beta": computed_beta,
        "current_price": round(prices.iloc[-1], 2),
        "data_points": len(prices),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def detect_significant_moves(
    ticker: str,
    period: str = "1y",
    threshold: float = 1.5,
    min_abs_return: float = 0.8,
) -> list:
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
        abs_ret = abs(ret * 100)
        if z_score >= threshold and abs_ret >= min_abs_return:
            events.append({
                "date": date,
                "return_pct": round(ret * 100, 2),
                "z_score": round(z_score, 1),
                "price": round(prices.loc[date], 2),
            })

    events.sort(key=lambda x: abs(x["return_pct"]), reverse=True)
    return events[:8]


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
"""
llm.py — Generate AI-powered fund summaries using Ollama (local).
"""

import requests


def _build_prompt(fund_data: dict) -> str:
    ticker = fund_data.get("ticker", "N/A")
    name = fund_data.get("name", "N/A")
    category = fund_data.get("category", "N/A")
    aum = fund_data.get("aum")
    expense_ratio = fund_data.get("expense_ratio")
    total_return = fund_data.get("total_return")
    ann_vol = fund_data.get("annualized_volatility")
    sharpe = fund_data.get("sharpe_ratio")
    max_dd = fund_data.get("max_drawdown")
    beta = fund_data.get("beta")
    dividend_yield = fund_data.get("dividend_yield")
    description = fund_data.get("description", "")

    return f"""You are a professional asset management analyst. Based on the data below, write a concise and insightful fund analysis note in 4-6 sentences.
Cover: the fund's investment objective, its risk/return profile, notable strengths or weaknesses, and a brief suitability comment.
Be factual, precise, and avoid marketing language.

--- FUND DATA ---
Ticker: {ticker}
Name: {name}
Category: {category}
AUM: {aum}
Expense Ratio: {expense_ratio}
Total Return (period): {total_return}%
Annualized Volatility: {ann_vol}%
Sharpe Ratio: {sharpe}
Max Drawdown: {max_dd}%
Beta: {beta}
Dividend Yield: {dividend_yield}
Description: {description[:500] if description else 'N/A'}
--- END ---

Write the analysis note now:"""


def generate_summary(fund_data: dict, provider: str = "Ollama") -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:4b",
                "prompt": _build_prompt(fund_data),
                "stream": False
            },
            timeout=60
        )
        return response.json()["response"]
    except Exception as e:
        return f"❌ Ollama error: {str(e)}"
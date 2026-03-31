"""
llm.py — Generate AI-powered fund analysis using Groq (Llama 3.1 70B).
"""
 
import os
from groq import Groq
 
 
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
 
    return f"""You are a senior fund analyst at a top asset management firm.
Write a professional investment analysis note for the fund below.
 
Structure your analysis in exactly 4 paragraphs:
 
1. FUND OVERVIEW: State the fund's objective, category, and size (AUM). 
   Mention the expense ratio and whether it is competitive for its category.
 
2. PERFORMANCE & RISK: Analyze the total return vs the level of risk taken.
   - If Sharpe > 1.0: strong risk-adjusted performance
   - If Sharpe 0.5-1.0: adequate compensation for risk
   - If Sharpe < 0.5: risk may not be adequately compensated
   Compare volatility and max drawdown to assess downside risk.
   Comment on the beta to indicate market sensitivity.
 
3. INCOME & SUITABILITY: Discuss the dividend yield if relevant.
   Specify which investor profile this fund suits:
   - Conservative (low vol, income-focused)
   - Balanced (moderate risk/return)
   - Growth-oriented (higher vol, capital appreciation)
   - Tactical/Satellite (thematic or concentrated exposure)
 
4. KEY CONSIDERATIONS: Mention 1-2 specific risks or watchpoints 
   (concentration risk, interest rate sensitivity, sector exposure, etc.)
   and one potential catalyst or strength going forward.
 
Be precise, use the actual numbers provided, and avoid generic marketing language.
Write in a professional but accessible tone. Keep each paragraph to 2-3 sentences.
 
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
 
 
def generate_summary(fund_data: dict, provider: str = "Groq") -> str:
    """Generate a fund analysis using Groq API (Llama 3.1 70B)."""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
 
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior fund analyst. Write precise, data-driven analysis notes."
                },
                {
                    "role": "user",
                    "content": _build_prompt(fund_data)
                }
            ],
            temperature=0.3,
            max_tokens=800,
        )
 
        return response.choices[0].message.content
 
    except Exception as e:
        return f"❌ Groq API error: {str(e)}"
"""
app.py — Fund Screener AI
Main Streamlit application.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

from utils.data import (
    fetch_price_history,
    build_fund_summary_dict,
    compute_metrics,
    format_aum,
    format_pct,
    PERIOD_MAP,
)
from utils.llm import generate_summary

load_dotenv()

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fund Screener AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 8px;
    }
    .metric-label { font-size: 0.8rem; color: #666; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 1.4rem; font-weight: 700; color: #1a1a2e; }
    .ai-box {
        background: linear-gradient(135deg, #f0f4ff 0%, #fafafa 100%);
        border: 1px solid #d0d7ff;
        border-radius: 12px;
        padding: 20px;
        margin-top: 12px;
        color: #1a1a2e;
    }
    }
    .tag {
        display: inline-block;
        background: #e8f0fe;
        color: #1967d2;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Fund Screener AI")
    st.markdown("*Powered by yfinance + LLMs*")
    st.divider()

    st.subheader("🔍 Search funds")
    raw_tickers = st.text_input(
        "Ticker(s) — separate with commas",
        value="SPY, QQQ, AGG",
        help="E.g. SPY, IWDA.L, ARKK"
    )
    tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]

    period_label = st.selectbox("📅 Analysis period", list(PERIOD_MAP.keys()), index=3)
    period = PERIOD_MAP[period_label]

    st.divider()
    st.subheader("🤖 AI Settings")
    llm_provider = st.selectbox("LLM provider", ["Groq (Llama 3.1 70B)"])
    auto_generate = st.toggle("Auto-generate AI summaries", value=False)

    st.divider()
    st.caption("Data from Yahoo Finance. For educational purposes only.")

# ─── Main ─────────────────────────────────────────────────────────────────────

st.header("📊 Fund Screener AI", divider="gray")

if not tickers:
    st.info("Enter at least one ticker in the sidebar to get started.")
    st.stop()

# ─── Load data ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_fund_data(ticker, period):
    return build_fund_summary_dict(ticker, period)

@st.cache_data(ttl=300)
def load_price_history(ticker, period):
    return fetch_price_history(ticker, period)

with st.spinner("Fetching market data..."):
    all_data = {t: load_fund_data(t, period) for t in tickers}
    all_prices = {t: load_price_history(t, period) for t in tickers}

# ─── Comparison Table ─────────────────────────────────────────────────────────

st.subheader("⚖️ Comparison Overview")

rows = []
for ticker, d in all_data.items():
    if "error" in d:
        rows.append({"Ticker": ticker, "Name": f"Error: {d['error']}"})
        continue
    rows.append({
        "Ticker": ticker,
        "Name": d.get("name", "N/A"),
        "Category": d.get("category", "N/A"),
        "AUM": format_aum(d.get("aum")),
        "Expense Ratio": format_pct(d.get("expense_ratio")),
        f"Return ({period_label})": f"{d.get('total_return', 'N/A')}%",
        "Ann. Volatility": f"{d.get('annualized_volatility', 'N/A')}%",
        "Sharpe Ratio": d.get("sharpe_ratio", "N/A"),
        "Max Drawdown": f"{d.get('max_drawdown', 'N/A')}%",
        "Beta": d.get("beta", "N/A"),
        "Dividend Yield": format_pct(d.get("dividend_yield")),
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

# CSV export
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Export to CSV", data=csv, file_name="fund_screener_results.csv", mime="text/csv")

# ─── Performance Chart ────────────────────────────────────────────────────────

st.subheader("📈 Historical Performance (normalized to 100)")

fig = go.Figure()
for ticker, hist in all_prices.items():
    if hist.empty:
        continue
    col = hist.columns[0]
    normalized = hist[col] / hist[col].iloc[0] * 100
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=normalized,
        mode="lines",
        name=ticker,
        line=dict(width=2),
    ))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Normalized Price (base 100)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400,
    margin=dict(l=0, r=0, t=20, b=0),
    plot_bgcolor="white",
)
fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
st.plotly_chart(fig, use_container_width=True)

# ─── Risk / Return Scatter ────────────────────────────────────────────────────

valid = {t: d for t, d in all_data.items() if "error" not in d and d.get("annualized_volatility") and d.get("total_return")}
if len(valid) >= 2:
    st.subheader("🎯 Risk / Return Profile")
    scatter_df = pd.DataFrame([
        {
            "Ticker": t,
            "Volatility (%)": d["annualized_volatility"],
            "Return (%)": d["total_return"],
            "Sharpe": d.get("sharpe_ratio", 0) or 0,
        }
        for t, d in valid.items()
    ])
    fig2 = px.scatter(
        scatter_df, x="Volatility (%)", y="Return (%)",
        text="Ticker", size=[20] * len(scatter_df),
        color="Sharpe", color_continuous_scale="RdYlGn",
        labels={"Sharpe": "Sharpe Ratio"},
    )
    fig2.update_traces(textposition="top center")
    fig2.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ─── Individual Fund Deep-Dives ───────────────────────────────────────────────

st.subheader("🔎 Fund Deep-Dives")
tabs = st.tabs([f"📄 {t}" for t in tickers])

for tab, ticker in zip(tabs, tickers):
    with tab:
        d = all_data[ticker]
        if "error" in d:
            st.error(f"Could not load data for {ticker}: {d['error']}")
            continue

        st.markdown(f"### {d.get('name', ticker)}")
        tags = []
        if d.get("category"): tags.append(d["category"])
        if d.get("asset_class"): tags.append(d["asset_class"])
        if d.get("currency"): tags.append(d["currency"])
        if tags:
            st.markdown(" ".join([f'<span class="tag">{t}</span>' for t in tags]), unsafe_allow_html=True)

        st.markdown("")

        # Key metrics in columns
        cols = st.columns(4)
        metrics = [
            ("Current Price", f"${d.get('current_price', 'N/A')}"),
            ("AUM", format_aum(d.get("aum"))),
            ("Expense Ratio", format_pct(d.get("expense_ratio"))),
            ("Dividend Yield", format_pct(d.get("dividend_yield"))),
            (f"Return ({period_label})", f"{d.get('total_return', 'N/A')}%"),
            ("Annualized Vol.", f"{d.get('annualized_volatility', 'N/A')}%"),
            ("Sharpe Ratio", str(d.get("sharpe_ratio", "N/A"))),
            ("Max Drawdown", f"{d.get('max_drawdown', 'N/A')}%"),
        ]
        for i, (label, value) in enumerate(metrics):
            with cols[i % 4]:
                st.metric(label, value)

        # Rolling volatility chart
        hist = all_prices.get(ticker, pd.DataFrame())
        if not hist.empty:
            col = hist.columns[0]
            returns = hist[col].pct_change().dropna()
            rolling_vol = returns.rolling(window=21).std() * (252 ** 0.5) * 100
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=rolling_vol.index, y=rolling_vol.values,
                mode="lines", fill="tozeroy",
                line=dict(color="#1f77b4", width=1.5),
                name="21-day Rolling Vol (%)",
            ))
            fig3.update_layout(
                title="21-Day Rolling Annualized Volatility",
                height=250,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="white",
            )
            fig3.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
            fig3.update_yaxes(showgrid=True, gridcolor="#f0f0f0", ticksuffix="%")
            st.plotly_chart(fig3, use_container_width=True)

        # AI Summary
        st.markdown("#### 🤖 AI Analysis Note")
        summary_key = f"summary_{ticker}_{llm_provider}"

        if auto_generate:
            if summary_key not in st.session_state:
                with st.spinner(f"Generating {llm_provider} analysis..."):
                    st.session_state[summary_key] = generate_summary(d, llm_provider)
            st.markdown(f'<div class="ai-box">{st.session_state[summary_key]}</div>', unsafe_allow_html=True)
        else:
            if st.button(f"Generate analysis with {llm_provider}", key=f"btn_{ticker}"):
                with st.spinner("Generating analysis..."):
                    st.session_state[summary_key] = generate_summary(d, llm_provider)
            if summary_key in st.session_state:
                st.markdown(f'<div class="ai-box">{st.session_state[summary_key]}</div>', unsafe_allow_html=True)
            else:
                st.caption("Click the button above to generate an AI-powered fund analysis.")

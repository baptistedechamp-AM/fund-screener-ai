"""
app.py — Fund Screener AI
Professional fund screening tool with AI-powered analysis.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from dotenv import load_dotenv

from utils.data import (
    fetch_price_history,
    build_fund_summary_dict,
    detect_significant_moves,
    format_aum,
    format_pct,
    PERIOD_MAP,
    PRESET_FUNDS,
)
from utils.llm import generate_summary, explain_market_events

load_dotenv()

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fund Screener AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Color helpers ───────────────────────────────────────────────────────────

COLORS = {
    "green": "#00D4AA",
    "red": "#FF6B6B",
    "blue": "#4DA6FF",
    "yellow": "#FFD93D",
    "gray": "#8892A0",
    "bg_card": "#1B1F2A",
    "bg_dark": "#0E1117",
    "text": "#E0E0E0",
    "text_muted": "#8892A0",
    "chart_colors": ["#00D4AA", "#4DA6FF", "#FFD93D", "#FF6B6B", "#C084FC", "#FB923C"],
}


def color_value(value, good_threshold=0, bad_threshold=None, reverse=False):
    if value is None or value == "N/A":
        return COLORS["gray"]
    try:
        v = float(value)
    except (ValueError, TypeError):
        return COLORS["gray"]
    if reverse:
        return COLORS["red"] if v > good_threshold else COLORS["green"]
    return COLORS["green"] if v >= good_threshold else COLORS["red"]


def metric_card(label, value, color=None, sub=None):
    c = color or COLORS["text"]
    html = f'<div class="metric-card"><div class="metric-label">{label}</div>'
    html += f'<div class="metric-value" style="color: {c}">{value}</div>'
    if sub:
        html += f'<div class="metric-sub">{sub}</div>'
    html += '</div>'
    return html


# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background: #1B1F2A;
        border-radius: 8px;
        padding: 14px 16px;
        border: 1px solid #2A2F3E;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #8892A0;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
    }
    .metric-sub {
        font-size: 0.7rem;
        color: #6B7280;
        margin-top: 2px;
    }
    .ai-box {
        background: #1B1F2A;
        border: 1px solid #2A3F5F;
        border-left: 3px solid #4DA6FF;
        border-radius: 8px;
        padding: 20px;
        margin-top: 12px;
        color: #E0E0E0;
        line-height: 1.7;
        font-size: 0.92rem;
    }
    .event-box {
        background: #1B1F2A;
        border: 1px solid #2A2F3E;
        border-left: 3px solid #FFD93D;
        border-radius: 8px;
        padding: 16px;
        margin-top: 8px;
        color: #E0E0E0;
        line-height: 1.6;
        font-size: 0.88rem;
    }
    .tag {
        display: inline-block;
        background: #2A2F3E;
        color: #8892A0;
        border-radius: 4px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 6px;
        border: 1px solid #3A3F4E;
    }
    .section-header {
        font-size: 0.78rem;
        font-weight: 600;
        color: #8892A0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #2A2F3E;
    }
    .updated-badge {
        font-size: 0.7rem;
        color: #6B7280;
        text-align: right;
        padding: 4px 0;
    }
    [data-testid="stMetricValue"] { font-size: 1.1rem; }
    .dataframe { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Fund Screener AI")
    st.caption("Professional ETF & fund analysis powered by AI")
    st.divider()

    st.markdown('<p class="section-header">Quick Presets</p>', unsafe_allow_html=True)
    preset_cols = st.columns(2)
    for i, (name, _tickers) in enumerate(PRESET_FUNDS.items()):
        with preset_cols[i % 2]:
            if st.button(name, use_container_width=True, key=f"preset_{name}"):
                st.session_state["active_tickers"] = ", ".join(PRESET_FUNDS[name])

    st.divider()

    st.markdown('<p class="section-header">Custom Search</p>', unsafe_allow_html=True)

    if "active_tickers" not in st.session_state:
        st.session_state["active_tickers"] = "SPY, QQQ, AGG"

    raw_tickers = st.text_input(
        "Tickers (comma-separated)",
        value=st.session_state["active_tickers"],
        help="Yahoo Finance tickers. European ETFs: add .L (London) or .DE (Frankfurt)",
    )
    tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]

    period_label = st.selectbox("Analysis period", list(PERIOD_MAP.keys()), index=4)
    period = PERIOD_MAP[period_label]

    st.divider()

    st.markdown('<p class="section-header">AI Analysis</p>', unsafe_allow_html=True)
    llm_provider = st.selectbox("Model", ["Groq (Llama 3.3 70B)"])
    auto_generate = st.toggle("Auto-generate on load", value=False)

    st.divider()
    st.caption("Data: Yahoo Finance + FMP — AI: Groq")
    st.caption("For educational purposes only. Not financial advice.")

# ─── Main Content ─────────────────────────────────────────────────────────────

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

@st.cache_data(ttl=300)
def load_events(ticker, period):
    return detect_significant_moves(ticker, period, threshold=2.0)

with st.spinner("Fetching market data..."):
    all_data = {t: load_fund_data(t, period) for t in tickers}
    all_prices = {t: load_price_history(t, period) for t in tickers}

valid_funds = {t: d for t, d in all_data.items() if "error" not in d}
error_funds = {t: d for t, d in all_data.items() if "error" in d}

if error_funds:
    for t, d in error_funds.items():
        st.warning(f"Could not load {t}: {d.get('error', 'Unknown error')}")

if not valid_funds:
    st.error("No valid funds found. Check your tickers.")
    st.stop()

# ─── Last updated badge ──────────────────────────────────────────────────────

first_fund = list(valid_funds.values())[0]
last_update = first_fund.get("last_updated", datetime.now().strftime("%Y-%m-%d %H:%M"))
st.markdown(f'<div class="updated-badge">Live data — Last refreshed: {last_update} — Cache: 5 min</div>', unsafe_allow_html=True)

# ─── Header cards ─────────────────────────────────────────────────────────────

header_cols = st.columns(len(valid_funds))
for i, (ticker, d) in enumerate(valid_funds.items()):
    with header_cols[i]:
        ret = d.get("total_return")
        ret_color = color_value(ret)
        price_display = d.get("current_price", "N/A")
        currency = d.get("currency", "USD")
        symbol = "£" if currency == "GBp" else "€" if currency == "EUR" else "$"
        ret_str = f"{'+' if ret and ret > 0 else ''}{ret}%" if ret else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{ticker}</div>
            <div class="metric-value" style="color: {COLORS['text']}">{symbol}{price_display}</div>
            <div style="color: {ret_color}; font-size: 0.9rem; font-weight: 600; margin-top: 4px;">
                {ret_str} ({period_label})
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")

# ─── Performance Chart with Event Annotations ────────────────────────────────

st.markdown('<p class="section-header">Performance (rebased to 100)</p>', unsafe_allow_html=True)

fig = go.Figure()
for i, (ticker, hist) in enumerate(all_prices.items()):
    if hist.empty:
        continue
    col = hist.columns[0]
    normalized = hist[col] / hist[col].iloc[0] * 100
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=normalized,
        mode="lines",
        name=ticker,
        line=dict(width=2, color=COLORS["chart_colors"][i % len(COLORS["chart_colors"])]),
    ))

# Collect top events across all tickers
all_events = []
for ticker_name in list(valid_funds.keys()):
    evts = load_events(ticker_name, period)
    hist = all_prices.get(ticker_name, pd.DataFrame())
    if evts and not hist.empty:
        col = hist.columns[0]
        norm = hist[col] / hist[col].iloc[0] * 100
        for evt in evts[:2]:
            if evt["date"] in norm.index:
                all_events.append({
                    "ticker": ticker_name,
                    "date": evt["date"],
                    "return_pct": evt["return_pct"],
                    "y_val": norm.loc[evt["date"]],
                })

# Keep top events spread at least 10 days apart
all_events.sort(key=lambda x: abs(x["return_pct"]), reverse=True)
shown_events = []
for evt in all_events:
    too_close = any(abs((evt["date"] - s["date"]).days) < 10 for s in shown_events)
    if not too_close:
        shown_events.append(evt)
    if len(shown_events) >= 4:
        break

for evt in shown_events:
    sign = "+" if evt["return_pct"] > 0 else ""
    evt_color = COLORS["green"] if evt["return_pct"] > 0 else COLORS["red"]
    fig.add_annotation(
        x=evt["date"],
        y=evt["y_val"],
        text=f"{evt['ticker']} {sign}{evt['return_pct']}%",
        showarrow=True, arrowhead=2, arrowsize=0.8,
        arrowcolor=evt_color,
        font=dict(size=10, color=evt_color, family="monospace"),
        bgcolor="#1B1F2A", bordercolor=evt_color,
        borderwidth=1, borderpad=3, ax=0, ay=-35,
    )

fig.update_layout(
    xaxis_title="",
    yaxis_title="",
    hovermode="x unified",
    legend=dict(
        orientation="h", yanchor="top", y=-0.08,
        xanchor="center", x=0.5, font=dict(size=12),
    ),
    height=420,
    margin=dict(l=0, r=0, t=10, b=60),
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117",
    font=dict(color="#8892A0", size=12),
    yaxis=dict(gridcolor="#1B1F2A", zerolinecolor="#1B1F2A"),
    xaxis=dict(gridcolor="#1B1F2A", zerolinecolor="#1B1F2A"),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ─── Market Events Explanation ────────────────────────────────────────────────

if shown_events:
    st.markdown('<p class="section-header">Market Events — AI Analysis</p>', unsafe_allow_html=True)
    events_key = f"events_{'_'.join([e['ticker'] + e['date'].strftime('%Y%m%d') for e in shown_events])}"

    if events_key not in st.session_state:
        if auto_generate:
            with st.spinner("Analyzing market events..."):
                st.session_state[events_key] = explain_market_events(shown_events)
        else:
            if st.button("Explain these market moves with AI", key="btn_events"):
                with st.spinner("Analyzing market events..."):
                    st.session_state[events_key] = explain_market_events(shown_events)

    if events_key in st.session_state:
        st.markdown(f'<div class="event-box">{st.session_state[events_key]}</div>', unsafe_allow_html=True)
    elif not auto_generate:
        st.caption("Click to get AI-powered explanations for the annotated market moves above.")

# ─── Comparison Table ─────────────────────────────────────────────────────────

st.markdown('<p class="section-header">Comparison Overview</p>', unsafe_allow_html=True)

rows = []
for ticker, d in valid_funds.items():
    er = d.get("expense_ratio")
    dy = d.get("dividend_yield")
    rows.append({
        "Ticker": ticker,
        "Name": d.get("name", "N/A"),
        "Category": d.get("category", "N/A"),
        "AUM": format_aum(d.get("aum")),
        "Expense Ratio": format_pct(er) if er else "—",
        "Return": f"{d.get('total_return', 'N/A')}%",
        "Volatility": f"{d.get('annualized_volatility', 'N/A')}%",
        "Sharpe": d.get("sharpe_ratio", "N/A"),
        "Sortino": d.get("sortino_ratio", "N/A"),
        "Max DD": f"{d.get('max_drawdown', 'N/A')}%",
        "Beta": d.get("beta", "N/A"),
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Export CSV", data=csv, file_name="fund_screener_results.csv", mime="text/csv")

# ─── Risk / Return Scatter ────────────────────────────────────────────────────

scatter_valid = {t: d for t, d in valid_funds.items() if d.get("annualized_volatility") and d.get("total_return")}
if len(scatter_valid) >= 2:
    st.markdown('<p class="section-header">Risk / Return Profile</p>', unsafe_allow_html=True)
    scatter_df = pd.DataFrame([
        {
            "Ticker": t,
            "Volatility (%)": d["annualized_volatility"],
            "Return (%)": d["total_return"],
            "Sharpe": d.get("sharpe_ratio", 0) or 0,
        }
        for t, d in scatter_valid.items()
    ])
    fig2 = px.scatter(
        scatter_df, x="Volatility (%)", y="Return (%)",
        text="Ticker", size=[20] * len(scatter_df),
        color="Sharpe", color_continuous_scale=["#FF6B6B", "#FFD93D", "#00D4AA"],
    )
    fig2.update_traces(textposition="top center", textfont=dict(size=13, color="#E0E0E0"))
    fig2.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#8892A0", size=12),
        xaxis=dict(gridcolor="#1B1F2A"),
        yaxis=dict(gridcolor="#1B1F2A"),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ─── Individual Fund Deep-Dives ───────────────────────────────────────────────

st.markdown('<p class="section-header">Fund Deep-Dives</p>', unsafe_allow_html=True)
tabs = st.tabs([f"{t}" for t in valid_funds.keys()])

for tab, (ticker, d) in zip(tabs, valid_funds.items()):
    with tab:
        st.markdown(f"#### {d.get('name', ticker)}")
        tags = []
        if d.get("category") and d["category"] != "N/A":
            tags.append(d["category"])
        if d.get("asset_class") and d["asset_class"] != "N/A":
            tags.append(d["asset_class"])
        if d.get("currency") and d["currency"] != "N/A":
            tags.append(d["currency"])
        if tags:
            st.markdown(" ".join([f'<span class="tag">{t}</span>' for t in tags]), unsafe_allow_html=True)

        st.markdown("")

        currency = d.get("currency", "USD")
        symbol = "£" if currency == "GBp" else "€" if currency == "EUR" else "$"
        er = d.get("expense_ratio")
        dy = d.get("dividend_yield")
        aum = d.get("aum")

        r1 = st.columns(4)
        with r1[0]:
            st.markdown(metric_card("Price", f"{symbol}{d.get('current_price', 'N/A')}"), unsafe_allow_html=True)
        with r1[1]:
            sub = "Not available for this fund" if not aum else None
            st.markdown(metric_card("AUM", format_aum(aum), COLORS["text"] if aum else COLORS["gray"], sub), unsafe_allow_html=True)
        with r1[2]:
            if er:
                er_color = color_value(er, good_threshold=0.005, reverse=True)
                st.markdown(metric_card("Expense Ratio", format_pct(er), er_color), unsafe_allow_html=True)
            else:
                st.markdown(metric_card("Expense Ratio", "—", COLORS["gray"], "Not reported by data source"), unsafe_allow_html=True)
        with r1[3]:
            if dy:
                st.markdown(metric_card("Dividend Yield", format_pct(dy), COLORS["blue"]), unsafe_allow_html=True)
            else:
                st.markdown(metric_card("Dividend Yield", "—", COLORS["gray"]), unsafe_allow_html=True)

        st.markdown("")

        r2 = st.columns(5)
        metrics_row2 = [
            ("Return", f"{d.get('total_return', 'N/A')}%", color_value(d.get("total_return"))),
            ("Volatility", f"{d.get('annualized_volatility', 'N/A')}%", color_value(d.get("annualized_volatility"), good_threshold=20, reverse=True)),
            ("Sharpe", str(d.get("sharpe_ratio", "N/A")), color_value(d.get("sharpe_ratio"), good_threshold=0.5)),
            ("Sortino", str(d.get("sortino_ratio", "N/A")), color_value(d.get("sortino_ratio"), good_threshold=0.5)),
            ("Max Drawdown", f"{d.get('max_drawdown', 'N/A')}%", color_value(d.get("max_drawdown"), good_threshold=-10, reverse=False)),
        ]
        for i, (label, value, clr) in enumerate(metrics_row2):
            with r2[i]:
                st.markdown(metric_card(label, value, clr), unsafe_allow_html=True)

        st.markdown("")

        r3 = st.columns(3)
        with r3[0]:
            beta = d.get("beta")
            beta_color = COLORS["green"] if beta and 0.8 <= float(beta) <= 1.2 else COLORS["yellow"] if beta else COLORS["gray"]
            st.markdown(metric_card("Beta (3Y)", beta if beta else "—", beta_color), unsafe_allow_html=True)
        with r3[1]:
            te = d.get("tracking_error")
            bench = d.get("benchmark_used", "")
            bench_label = f"vs {bench}" if bench and bench.upper() != ticker.upper() else ""
            te_display = f"{te}%" if te else "—"
            st.markdown(metric_card(f"Tracking Error {bench_label}", te_display, COLORS["blue"] if te else COLORS["gray"]), unsafe_allow_html=True)
        with r3[2]:
            dp = d.get("data_points")
            st.markdown(metric_card("Data Points", dp if dp else "N/A", COLORS["gray"]), unsafe_allow_html=True)

        st.markdown("")

        hist = all_prices.get(ticker, pd.DataFrame())
        if not hist.empty:
            col = hist.columns[0]
            returns = hist[col].pct_change().dropna()
            rolling_vol = returns.rolling(window=21).std() * (252 ** 0.5) * 100

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=rolling_vol.index, y=rolling_vol.values,
                mode="lines", fill="tozeroy",
                line=dict(color=COLORS["blue"], width=1.5),
                fillcolor="rgba(77, 166, 255, 0.1)",
                name="21d Rolling Vol",
            ))

            fund_events = load_events(ticker, period)
            for evt in fund_events[:3]:
                evt_date = evt["date"]
                if evt_date in rolling_vol.index:
                    y_val = rolling_vol.loc[evt_date]
                    if pd.notna(y_val):
                        sign = "+" if evt["return_pct"] > 0 else ""
                        evt_color = COLORS["green"] if evt["return_pct"] > 0 else COLORS["red"]
                        fig3.add_annotation(
                            x=evt_date, y=y_val,
                            text=f"{sign}{evt['return_pct']}%",
                            showarrow=True, arrowhead=2, arrowsize=0.8,
                            arrowcolor=evt_color,
                            font=dict(size=9, color=evt_color),
                            bgcolor="#1B1F2A", bordercolor=evt_color,
                            borderwidth=1, borderpad=2, ax=0, ay=-25,
                        )

            fig3.update_layout(
                title=dict(text="21-Day Rolling Annualized Volatility", font=dict(size=13, color="#8892A0")),
                height=220,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="#0E1117",
                paper_bgcolor="#0E1117",
                font=dict(color="#8892A0", size=11),
                yaxis=dict(gridcolor="#1B1F2A", ticksuffix="%"),
                xaxis=dict(gridcolor="#1B1F2A"),
                showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

        # AI Analysis
        st.markdown('<p class="section-header">AI Analysis Note</p>', unsafe_allow_html=True)
        summary_key = f"summary_{ticker}_{llm_provider}"

        if auto_generate:
            if summary_key not in st.session_state:
                with st.spinner("Generating analysis..."):
                    st.session_state[summary_key] = generate_summary(d, llm_provider)
            st.markdown(f'<div class="ai-box">{st.session_state[summary_key]}</div>', unsafe_allow_html=True)
        else:
            if st.button("Generate AI analysis", key=f"btn_{ticker}"):
                with st.spinner("Analyzing fund data..."):
                    st.session_state[summary_key] = generate_summary(d, llm_provider)
            if summary_key in st.session_state:
                st.markdown(f'<div class="ai-box">{st.session_state[summary_key]}</div>', unsafe_allow_html=True)
            else:
                st.caption("Click to generate a professional AI-powered fund analysis.")
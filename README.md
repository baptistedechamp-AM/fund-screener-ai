# 📊 Fund Screener AI

An AI-powered fund and ETF screening application built with Python and Streamlit.
Designed to replicate the kind of quantitative analysis workflow used in asset management.

## 🚀 Features

- **Multi-fund analysis** — Search and compare any ETF or fund by ticker (SPY, QQQ, AGG, etc.)
- **Key metrics** — AUM, expense ratio, total return, annualized volatility, Sharpe ratio, max drawdown, beta, dividend yield
- **Interactive charts** — Price performance visualization over custom time periods
- **AI-generated analysis notes** — Local LLM (Ollama) produces concise, professional fund commentary

## 🛠️ Tech Stack

| Tool | Role |
|---|---|
| Python | Core language |
| Streamlit | Web interface |
| yfinance | Market data (Yahoo Finance) |
| Ollama + Gemma 3 | Local AI model for analysis generation |

## ⚙️ Installation
```bash
# Clone the repository
git clone https://github.com/baptistedechamp-AM/fund-screener-ai.git
cd fund-screener-ai

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Ollama and pull the model
# Download Ollama at https://ollama.com
ollama pull gemma3:4b

# Run the app
streamlit run app.py
```

## 📈 Example Output

The app generates professional analysis notes such as:

> *"The State Street SPDR S&P 500 ETF Trust (SPY) tracks the S&P 500, representing a
> large-blend equity strategy. Over the past period, SPY delivered a total return of 14.27%,
> with an annualized volatility of 18.91% and a Sharpe Ratio of 0.59..."*

## 👤 Author

Baptiste Dechamp — Finance student, aspiring asset management analyst

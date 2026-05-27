# AIStock 📈🤖

AIStock is an AI-powered stock market analysis and backtesting web application built using Python and Streamlit. The project allows users to analyze US and Indian stocks, visualize market trends, compare historical performance, and generate AI-based stock insights using machine learning concepts.

The application supports multiple stock markets and fetches financial data using free public sources such as Yahoo Finance and Stooq without requiring paid APIs.

---
## 🎥 Demo Video

Watch the complete project demonstration here:

[![AiStock Demo](https://img.shields.io/badge/Watch-Demo_Video-red?style=for-the-badge&logo=youtube)](https://youtu.be/R_ecGF3B0Tc)

---


# Features 🚀

- 📊 Interactive stock market visualization
- 🇺🇸 US stock market support
- 🇮🇳 Indian stock market support
- 📈 Historical OHLCV chart analysis
- 🤖 AI-based stock trend analysis
- 🧠 Machine Learning integration
- 🌙 Dark / Light theme mode
- 🔍 Stock comparison functionality
- 📉 Technical indicators & trend analysis
- 📋 Chart-to-table numerical conversion
- 📥 CSV & PNG export/download support
- ⚡ Fast interactive Streamlit dashboard
- 🆓 No paid API required

---

# Tech Stack 🛠️

## Frontend
- Streamlit
- HTML
- CSS

## Backend
- Python

## Libraries Used
- pandas
- numpy
- matplotlib
- plotly
- scikit-learn
- yfinance
- streamlit
- ta
- requests

---

# Data Sources 📡

This project currently uses free public financial market sources:

- Yahoo Finance
- Stooq Market Data

No API keys are required for the current version of the project.

---

# Major Challenge Solved 🧩

Initially, the project successfully analyzed US stocks such as AAPL, but Indian stock market analysis failed because Yahoo Finance requires exchange-specific ticker formats like `.NS` and `.BO`.

To solve this issue, ticker normalization and preprocessing logic were implemented. Now, users can directly enter stock names such as `Reliance` or `TCS`, and the system automatically converts them into the correct market-compatible format internally before analysis.

Additional improvements include:
- Fallback mechanisms between data sources
- Improved error handling
- Market-date validation for weekends/holidays
- Better cross-market compatibility

---

# Prediction Horizon Experimentation 📉

The project was tested using multiple forecasting horizons, including 10-day and 5-day prediction windows.

Results showed that:
- 5-day horizon produced more stable performance
- Reduced losses compared to the 10-day configuration
- Improved drawdown control
- Generated better backtesting metrics during volatile market conditions

This experimentation helped optimize the final strategy configuration used in the project.

---

# Project Structure 📂

```bash
AIStock/
│
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── data.py
│   ├── model.py
│   ├── indicators.py
│   └── utils.py
│
├── assets/
├── models/
├── notebooks/
└── .venv/
```

---

# Installation ⚙️

## 1. Clone Repository

```bash
git clone https://github.com/MinalMaurya/ai-stock-market-analysis.git
cd ai-stock-market-analysis
```

---

## 2. Create Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Project ▶️

```bash
streamlit run app.py
```

---

# Supported Markets 🌍

| Market | Example |
|---|---|
| US Stocks | AAPL, TSLA, MSFT |
| Indian Stocks | RELIANCE.NS, TCS.NS |

---

# Machine Learning Features 🧠

- Stock trend prediction
- Time-series forecasting
- Technical analysis
- Moving averages
- Volatility analysis
- AI-generated insights
- Backtesting & strategy evaluation

---

# Future Improvements 🔮

- Live stock streaming
- News sentiment analysis
- Portfolio management
- User authentication
- Advanced ML/DL models
- API-based live trading integration
- Cloud deployment support

---

# Known Issues ⚠️

- Minor dark/light mode UI inconsistencies
- Some Streamlit components may not fully follow custom themes
- Public market data sources may occasionally face rate limits

---

# Why This Project? 💡

This project was developed to explore:
- AI/ML in finance
- Stock market analytics
- Financial forecasting
- Data visualization
- Real-world machine learning applications
- Risk-analysis based backtesting

---

# License 📄

This project is developed for educational and learning purposes only.

---

# Author 👨‍💻

## Minal Maurya

- AI/ML Enthusiast
- Python Developer
- Interested in Financial Analytics and Intelligent Systems

---

# GitHub Topics 🏷️

```text
python
machine-learning
stock-market
streamlit
ai
finance
data-science
stock-prediction
yfinance
india-stock-market
```

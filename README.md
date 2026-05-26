# AIStock 📈🤖

AIStock is an AI-powered stock market analysis and prediction web application built using Python and Streamlit. The project allows users to analyze US and Indian stocks, visualize market trends, compare historical performance, and generate AI-based stock insights using machine learning models.

The application supports multiple stock markets and fetches financial data using free public sources like Yahoo Finance and Stooq without requiring paid APIs.

---

# Features 🚀

- 📊 Real-time stock market visualization
- 🇺🇸 US stock support
- 🇮🇳 Indian stock support
- 📈 Historical OHLCV chart analysis
- 🤖 AI-based stock prediction
- 🧠 Machine Learning integration
- 🌙 Dark / Light theme mode
- 🔍 Stock comparison functionality
- 📉 Technical indicators and trends
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
git clone https://github.com/your-username/AIStock.git
cd AIStock
```

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

---


# Future Improvements 🔮

- Live stock streaming
- News sentiment analysis
- Portfolio management
- User authentication
- Advanced ML/DL models
- API-based live trading integration
- Deployment on cloud platforms

---

# Known Issues ⚠️

- Minor dark/light mode UI inconsistencies
- Some Streamlit components may not fully follow custom themes
- Public data sources may occasionally face rate limits

---

# Why This Project? 💡

This project was developed to explore:
- AI/ML in finance
- Stock market analytics
- Financial forecasting
- Data visualization
- Real-world machine learning applications

---

# License 📄

This project is for educational and learning purposes.

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

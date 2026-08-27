# European Indices AI Forecasting Terminal

An end-to-end Streamlit dashboard for algorithmic trading analytics and price forecasting across European financial indices. The application combines live market data, technical feature engineering, XGBoost, and a Temporal Fusion Transformer (TFT).

## Features

- Live market data through Yahoo Finance using `yfinance`
- XGBoost single-step next-close prediction
- Temporal Fusion Transformer multi-horizon forecasting
- Technical indicators including RSI, MACD, Bollinger Bands, SMA, EMA, and ATR
- Lagged returns and close-price features
- Interactive Plotly candlestick charts
- Local CSV upload support for custom inference
- Intraday Yahoo requests constrained to supported lookback periods

## Supported Assets

The trained models are intended for these seven European indices:

- `DE40` - Germany
- `FRA40` - France
- `UK100` - United Kingdom
- `EUSTX50` - Euro Stoxx 50
- `ESP35` - Spain
- `IT40` - Italy
- `SWI20` - Switzerland

Custom CSV uploads can contain other assets, but predictions outside the training assets may be unreliable.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 through 3.13 (`pandas-ta` currently does not support Python 3.14)
- Internet access for live Yahoo Finance data

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/alihaider1001/Forex.git
cd Forex
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.

## Using Live Data

1. Select `Live Yahoo Finance` in the sidebar.
2. Select an index and timeframe.
3. Select either the XGBoost or TFT model.
4. Select `Generate Forecast`.

Yahoo Finance limits historical intraday data. The application uses a maximum 60-day period for intraday intervals and retries with a shorter period when Yahoo returns an empty response.

## Using a Local CSV

1. Select `Local CSV` in the sidebar.
2. Upload a `.csv` file.
3. Select the model and click `Generate Forecast`.

Raw uploaded data should include these columns:

```text
datetime,open,high,low,close,volume
```

The feature pipeline calculates the required indicators and lag features before inference. At least 70 usable rows are recommended for TFT inference; rows containing incomplete indicator values are removed.

## Project Files

- `app.py` - Streamlit user interface and inference workflows
- `live_pipeline.py` - OHLCV feature engineering for live and uploaded data
- `requirements.txt` - Python dependencies
- `xgboost_model.json` - XGBoost model artifact
- `tft_model.ckpt` - TFT checkpoint
- `*_features.csv` - Feature-enriched historical datasets
- `*_M1_*.csv` - Raw minute-level market data

## Disclaimer

This project is for research and educational purposes. Forecasts are estimates, not financial advice. Market data can be delayed, incomplete, or unavailable, and model predictions can be inaccurate. Do not trade with money you cannot afford to lose.

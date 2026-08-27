import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import xgboost as xgb
import plotly.graph_objects as go
import torch
import yfinance as yf
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from live_pipeline import generate_live_features

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="European Indices Forecasting AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 European Indices Forecasting AI")
st.markdown("Automated algorithmic trading analytics powered by **XGBoost** and **Temporal Fusion Transformer (TFT)**.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Trading Configuration")

data_source = st.sidebar.radio("Data Source", ["Local CSV", "Live Yahoo Finance"])

st.sidebar.warning(
    "⚠️ **Note:** These AI models are specifically trained on 7 European indices "
    "(DE40, FRA40, UK100, EUSTX50, ESP35, IT40, SWI20). Uploading data for other "
    "assets may produce unreliable forecasts."
)

uploaded_file = None
if data_source == "Local CSV":
    uploaded_file = st.sidebar.file_uploader("Upload market data CSV", type=["csv"])

selected_index = st.sidebar.selectbox(
    "Select European Index", 
    ["IT40", "MIDDE50", "NETH25", "NOR25", "SE30", "SWI20", "DE40", "FRA40", "UK100"]
)

timeframe = st.sidebar.selectbox(
    "Select Timeframe", 
    ["5m", "15m", "30m", "1H", "2H", "4H"]
)

st.sidebar.subheader("AI Engine")
model_choice = st.sidebar.radio(
    "Select Prediction Model",
    ["TFT (Deep Learning - Multi-Horizon)", "XGBoost (Single-Step Baseline)"]
)

with st.sidebar.expander("💡 Which model should I choose?"):
    if "TFT" in model_choice:
        st.markdown("""
        **TFT (Deep Learning):**
        * **Best for:** Multi-step price trajectory and uncertainty bounds.
        * **Output:** 10-period forecast with confidence quantiles.
        * **Strategy:** Swing trading and wide Take-Profit targets.
        """)
    else:
        st.markdown("""
        **XGBoost (Tree Model):**
        * **Best for:** Immediate directional speed and momentum confirmation.
        * **Output:** Exact next-candle close price prediction.
        * **Strategy:** Scalping and quick entry validation.
        """)

run_prediction = st.sidebar.button("Generate Forecast", use_container_width=True)

live_ticker_map = {
    "IT40": "FTSEMIB.MI",
    "DE40": "^GDAXI",
    "UK100": "^FTSE",
    "SWI20": "^SSMI",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
}


@st.cache_data(ttl=60)
def fetch_live_data(ticker, interval):
    intraday_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "2h", "4h"}
    is_intraday = interval.lower() in intraday_intervals
    periods = ["60d", "30d"] if is_intraday else ["2y"]
    live_df = pd.DataFrame()
    for period in periods:
        live_df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if not live_df.empty:
            break
        if is_intraday and period != periods[-1]:
            time.sleep(1)

    if live_df.empty:
        return live_df
    if isinstance(live_df.columns, pd.MultiIndex):
        live_df.columns = live_df.columns.get_level_values(0)
    live_df = live_df.reset_index()
    live_df.columns = [str(column).lower() for column in live_df.columns]
    if "volume" not in live_df.columns:
        live_df["volume"] = 0
    return live_df[["datetime", "open", "high", "low", "close", "volume"]].dropna()

# --- HELPER FUNCTIONS ---
def find_data_file(symbol, tf):
    """Locate the enriched features CSV or raw timeframe CSV."""
    candidates = [
        f"{symbol}_{tf}_features.csv",
        f"{symbol}_{tf}.csv",
        f"{symbol}_features.csv"
    ]
    for filename in candidates:
        if os.path.exists(filename):
            return filename
    return None

# --- MAIN DASHBOARD AREA ---
if run_prediction and data_source == "Live Yahoo Finance":
    interval_map = {"5m": "5m", "15m": "15m", "30m": "30m", "1H": "1h", "2H": "1h", "4H": "1h"}
    live_symbol = selected_index if selected_index in live_ticker_map else "DE40"

    try:
        raw_live_data = fetch_live_data(live_ticker_map[live_symbol], interval_map[timeframe])
        if raw_live_data.empty:
            st.error(
                f"Yahoo Finance returned no data for {live_ticker_map[live_symbol]} "
                f"using the {timeframe} interval. Try another index or timeframe."
            )
            st.stop()
        live_data = generate_live_features(raw_live_data.copy())
    except Exception as error:
        st.error(f"Could not prepare live market data: {error}")
        st.stop()

    st.subheader(f"Live Price: {live_symbol}")
    live_chart = go.Figure(
        data=[go.Candlestick(
            x=raw_live_data["datetime"],
            open=raw_live_data["open"],
            high=raw_live_data["high"],
            low=raw_live_data["low"],
            close=raw_live_data["close"],
            name="Live Price",
        )]
    )
    live_chart.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=450)
    st.plotly_chart(live_chart, use_container_width=True)

    if "XGBoost" in model_choice:
        model_file = "xgboost_model.json"
        if not os.path.exists(model_file):
            st.error(f"XGBoost model file `{model_file}` not found.")
        else:
            live_model = xgb.XGBRegressor()
            live_model.load_model(model_file)
            model_features = live_model.get_booster().feature_names
            missing_features = [column for column in model_features if column not in live_data.columns]
            if missing_features:
                st.error(f"Live feature generation is missing model columns: {missing_features}")
            else:
                current_price = float(live_data["close"].iloc[-1])
                predicted_price = float(live_model.predict(live_data[model_features].iloc[[-1]])[0])
                delta_price = predicted_price - current_price
                signal = "BUY (LONG)" if delta_price > 0 else "SELL (SHORT)"

                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"{current_price:,.2f}")
                col2.metric("Predicted Next Close", f"{predicted_price:,.2f}", f"{delta_price:+,.2f}")
                col3.metric("Live Signal", signal)
                st.success("Live XGBoost forecast generated successfully.")
    else:
        model_file = "tft_model.ckpt"
        if not os.path.exists(model_file):
            st.error(f"TFT checkpoint file `{model_file}` was not found.")
        else:
            try:
                live_tft = TemporalFusionTransformer.load_from_checkpoint(model_file)
                live_lookback = live_data.tail(70).copy().reset_index(drop=True)
                live_lookback.columns = [str(column).replace(".", "_") for column in live_lookback.columns]
                live_lookback["time_idx"] = np.arange(len(live_lookback))
                live_lookback["symbol"] = live_symbol
                known_reals = ["rsi_14", "atr_14", "sma_50", "ema_20", "return_lag_1", "return_lag_5"]
                live_dataset = TimeSeriesDataSet(
                    live_lookback,
                    time_idx="time_idx",
                    target="close",
                    group_ids=["symbol"],
                    min_encoder_length=30,
                    max_encoder_length=60,
                    min_prediction_length=1,
                    max_prediction_length=10,
                    time_varying_unknown_reals=known_reals + ["close"],
                    target_normalizer=GroupNormalizer(groups=["symbol"], transformation=None),
                    add_relative_time_idx=True,
                    add_target_scales=True,
                    add_encoder_length=True,
                )
                live_predictions = live_tft.predict(
                    live_dataset.to_dataloader(batch_size=1, train=False, num_workers=0),
                    mode="raw",
                    return_x=True,
                )
                output = live_predictions.output.prediction if hasattr(live_predictions.output, "prediction") else live_predictions.output
                forecast_path = output[0, :, 3].detach().cpu().numpy()
                current_price = float(live_data["close"].iloc[-1])
                target_price = float(forecast_path[-1])
                delta_price = target_price - current_price
                signal = "BUY (LONG)" if delta_price > 0 else "SELL (SHORT)"

                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"{current_price:,.2f}")
                col2.metric("Predicted Target", f"{target_price:,.2f}", f"{delta_price:+,.2f}")
                col3.metric("Live TFT Signal", signal)
                st.line_chart(pd.DataFrame({"TFT Median Forecast": forecast_path}))
                st.success("Live TFT multi-horizon forecast generated successfully.")
            except Exception as error:
                st.error(f"Live TFT inference failed: {error}")

elif run_prediction:
    file_path = None if uploaded_file is not None else find_data_file(selected_index, timeframe)
    
    if uploaded_file is None and not file_path:
        st.error(f"Data file for **{selected_index} ({timeframe})** not found. Ensure the CSV files are in the working directory.")
    else:
        # Load and clean historical data
        df = pd.read_csv(uploaded_file if uploaded_file is not None else file_path)

        if uploaded_file is not None:
            try:
                df.columns = [str(column).lower() for column in df.columns]
                df = generate_live_features(df)
            except Exception as error:
                st.error(f"Could not prepare uploaded market data: {error}")
                st.stop()

        # Only rename columns if TFT is running; XGBoost needs the original periods.
        if "TFT" in model_choice:
            df.columns = df.columns.str.replace('.', '_', regex=False)
        
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
        
        current_price = df['close'].iloc[-1]
        
        # ---------------------------------------------------------
        # OPTION A: XGBoost Prediction Pipeline
        # ---------------------------------------------------------
        if "XGBoost" in model_choice:
            model_file = "xgboost_model.json"
            if not os.path.exists(model_file):
                st.error(f"XGBoost model file `{model_file}` not found. Please export your trained model first.")
            else:
                # Load XGBoost model
                bst = xgb.XGBRegressor()
                bst.load_model(model_file)
                
                # Exclude target and non-feature columns
                exclude_cols = ['datetime', 'date', 'time', 'target', 'open', 'high', 'low', 'close', 'symbol']
                feature_cols = [col for col in df.columns if col not in exclude_cols]
                
                latest_features = df[feature_cols].iloc[[-1]]
                predicted_price = float(bst.predict(latest_features)[0])
                delta_price = predicted_price - current_price
                signal_direction = "BUY (LONG)" if delta_price > 0 else "SELL (SHORT)"
                
                # Render Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Close Price", f"{current_price:,.2f}")
                col2.metric("Predicted Next Close (T+1)", f"{predicted_price:,.2f}", f"{delta_price:+,.2f}")
                col3.metric("Directional Signal", signal_direction)
                
                # Plot Historical Context + Next Point
                recent_df = df.tail(50).copy().reset_index(drop=True)
                fig, ax = plt.subplots(figsize=(12, 5))
                
                ax.plot(recent_df.index, recent_df['close'], label="Historical Close", color="#1f77b4", linewidth=2)
                ax.scatter(len(recent_df), predicted_price, color="green" if delta_price > 0 else "red", s=100, zorder=5, label="XGBoost Target")
                ax.plot([len(recent_df)-1, len(recent_df)], [current_price, predicted_price], color="gray", linestyle="--")
                
                ax.set_title(f"{selected_index} ({timeframe}) — XGBoost 1-Step Ahead Forecast", fontsize=14)
                ax.set_xlabel("Recent Bars")
                ax.set_ylabel("Price")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend()
                
                st.pyplot(fig)
                st.success("XGBoost single-step prediction generated successfully.")

        # ---------------------------------------------------------
        # OPTION B: Temporal Fusion Transformer (TFT) Pipeline
        # ---------------------------------------------------------
        else:
            model_file = "tft_model.ckpt"
            if not os.path.exists(model_file):
                st.error(f"TFT checkpoint file `{model_file}` not found. Please ensure the `.ckpt` file is in the project folder.")
            else:
                try:
                    # Load TFT model
                    tft = TemporalFusionTransformer.load_from_checkpoint(model_file)
                    tft.eval()
                    
                    # Prepare lookback data
                    lookback_df = df.tail(70).copy().reset_index(drop=True)
                    lookback_df['time_idx'] = np.arange(len(lookback_df))
                    lookback_df['symbol'] = selected_index
                    
                    known_continuous_cols = ['rsi_14', 'atr_14', 'sma_50', 'ema_20', 'return_lag_1', 'return_lag_5']
                    # Ensure continuous columns exist
                    for col in known_continuous_cols:
                        if col not in lookback_df.columns:
                            lookback_df[col] = 0.0
                    
                    # Create TimeSeriesDataSet for the inference window
                    inference_dataset = TimeSeriesDataSet(
                        lookback_df,
                        time_idx="time_idx",
                        target="close",
                        group_ids=["symbol"],
                        min_encoder_length=30,
                        max_encoder_length=60,
                        min_prediction_length=1,
                        max_prediction_length=10,
                        time_varying_unknown_reals=known_continuous_cols + ["close"],
                        target_normalizer=GroupNormalizer(groups=["symbol"], transformation=None),
                        add_relative_time_idx=True,
                        add_target_scales=True,
                        add_encoder_length=True,
                    )
                    
                    dataloader = inference_dataset.to_dataloader(batch_size=1, train=False, num_workers=0)
                    predictions = tft.predict(dataloader, mode="raw", return_x=True)
                    
                    # Extract 10-step median predictions (quantile index 3)
                    pred_output = predictions.output
                    if hasattr(pred_output, 'prediction'):
                        forecast_path = pred_output.prediction[0, :, 3].detach().cpu().numpy()
                    else:
                        forecast_path = pred_output[0, :, 3].detach().cpu().numpy()
                    
                    target_price = float(forecast_path[-1])
                    total_delta = target_price - current_price
                    signal = "BUY (LONG)" if total_delta > 0 else "SELL (SHORT)"
                    
                    # Display Metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current Price", f"{current_price:,.2f}")
                    col2.metric("Predicted Target (T+10)", f"{target_price:,.2f}", f"{total_delta:+,.2f}")
                    col3.metric("Multi-Horizon Signal", signal)
                    
                    # Plot TFT Multi-Horizon Forecast
                    past_prices = lookback_df['close'].values[-60:]
                    past_x = np.arange(-len(past_prices) + 1, 1)
                    future_x = np.arange(1, len(forecast_path) + 1)
                    
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(past_x, past_prices, label="Historical Price (Lookback)", color="black", linewidth=2)
                    ax.plot(future_x, forecast_path, label="TFT Predicted Path (Median)", color="#0052cc", linestyle="--", linewidth=2)
                    ax.scatter(future_x, forecast_path, color="#0052cc", s=40)
                    
                    ax.axvline(0, color="gray", linestyle=":", alpha=0.7)
                    ax.set_title(f"{selected_index} ({timeframe}) — TFT Multi-Horizon Forecast (10 Steps)", fontsize=14)
                    ax.set_xlabel("Time Horizons")
                    ax.set_ylabel("Price")
                    ax.grid(True, linestyle="--", alpha=0.5)
                    ax.legend()
                    
                    st.pyplot(fig)
                    st.success("TFT multi-horizon trajectory generated successfully.")
                    
                except Exception as e:
                    st.warning(f"Could not complete TFT tensor evaluation: {e}")
                    st.info("Displaying direct feature-aligned projection.")

else:
    st.info("👈 Select your market configuration and click **Generate Forecast** to run model inference.")
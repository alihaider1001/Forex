import pandas as pd
import pandas_ta_classic as ta


def generate_live_features(df):
    """Generate the same indicators and lag features used during training."""
    required_columns = {"open", "high", "low", "close", "volume"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Live data is missing columns: {sorted(missing_columns)}")

    features = df.copy()
    features["tickvol"] = features["volume"]
    features["rsi_14"] = ta.rsi(features["close"], length=14)
    features = pd.concat(
        [features, ta.macd(features["close"], fast=12, slow=26, signal=9)],
        axis=1,
    )
    features = pd.concat(
        [features, ta.bbands(features["close"], length=20, std=2)],
        axis=1,
    )
    features = features.rename(
        columns={
            f"{prefix}_20_2.0": f"{prefix}_20_2.0_2.0"
            for prefix in ["BBL", "BBM", "BBU", "BBB", "BBP"]
        }
    )
    features["atr_14"] = ta.atr(
        features["high"], features["low"], features["close"], length=14
    )
    features["sma_50"] = ta.sma(features["close"], length=50)
    features["ema_20"] = ta.ema(features["close"], length=20)

    for lag in [1, 2, 3, 5, 10]:
        features[f"return_lag_{lag}"] = features["close"].pct_change(lag)
        features[f"close_lag_{lag}"] = features["close"].shift(lag)

    return features.dropna().reset_index(drop=True)

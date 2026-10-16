"""Data ingestion, feature engineering, and sequence generation module.

Provides mathematical indicator implementations for quantitative forecasting:
- Logarithmic returns
- Rolling sample volatility
- Stationarized EWMA price deviation
- Acceleration momentum (second discrete derivative of log price)
- PyTorch Dataset and DataLoader generation with strict time-series splits
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset
import yfinance as yf


# ==============================================================================
# 1. Mathematical Indicator Calculations
# ==============================================================================

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute continuous logarithmic returns: r_t = ln(P_t) - ln(P_{t-1}).

    Args:
        prices: Series of asset prices (typically Close or Adjusted Close).

    Returns:
        Series of logarithmic returns.
    """
    return np.log(prices / prices.shift(1))


def compute_rolling_volatility(
    log_returns: pd.Series,
    window: int = 20,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    """Compute rolling historical sample standard deviation of log returns.

    Args:
        log_returns: Series of logarithmic returns.
        window: Lookback window size w.
        annualize: Whether to multiply by sqrt(periods_per_year).
        periods_per_year: Trading periods per year (e.g. 252 for daily, 252*390 for 1-min).

    Returns:
        Series of rolling volatility values.
    """
    rolling_vol = log_returns.rolling(window=window).std()
    if annualize:
        rolling_vol = rolling_vol * np.sqrt(periods_per_year)
    return rolling_vol


def compute_ewma_deviation(prices: pd.Series, span: int = 20) -> pd.Series:
    """Compute stationarized price deviation relative to EWMA.

    delta_EWMA,t = (P_t - EWMA_t) / EWMA_t

    Args:
        prices: Series of asset prices.
        span: EWMA span parameter S (decay factor alpha = 2 / (S + 1)).

    Returns:
        Series of fractional deviations from the EWMA.
    """
    ewma = prices.ewm(span=span, adjust=False).mean()
    return (prices - ewma) / ewma


def compute_acceleration_momentum(
    prices: pd.Series,
    lag: int = 5,
    smooth_span: Optional[int] = 3,
) -> pd.Series:
    """Compute custom acceleration momentum factor (second discrete derivative).

    First derivative (velocity):
        v_t = (ln(P_t) - ln(P_{t-k})) / k
    Second derivative (acceleration):
        a_t = (v_t - v_{t-k}) / k = (ln(P_t) - 2*ln(P_{t-k}) + ln(P_{t-2k})) / k^2

    Args:
        prices: Series of asset prices.
        lag: Discrete time step lag k.
        smooth_span: Optional EWMA span to filter microstructure noise before differentiation.

    Returns:
        Series of discrete acceleration values.
    """
    if smooth_span is not None and smooth_span > 1:
        eval_prices = prices.ewm(span=smooth_span, adjust=False).mean()
    else:
        eval_prices = prices

    log_prices = np.log(eval_prices)
    velocity = (log_prices - log_prices.shift(lag)) / lag
    acceleration = (velocity - velocity.shift(lag)) / lag
    return acceleration


# ==============================================================================
# 2. Data Ingestion & Feature Engineering Pipeline
# ==============================================================================

def generate_synthetic_ohlcv(
    ticker: str = "SYNTH",
    start: str = "2018-01-01",
    end: str = "2024-01-01",
    initial_price: float = 300.0,
    annual_drift: float = 0.08,
    annual_vol: float = 0.18,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data using Geometric Brownian Motion.

    Useful for offline testing or when external API connectivity is blocked.
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start, end=end, freq="B")  # Business days
    n = len(dates)
    dt = 1.0 / 252.0

    # Geometric Brownian Motion log returns
    drift = (annual_drift - 0.5 * (annual_vol**2)) * dt
    shocks = annual_vol * np.sqrt(dt) * np.random.standard_normal(n)
    log_returns = drift + shocks

    close_prices = initial_price * np.exp(np.cumsum(log_returns))
    open_prices = close_prices * (1.0 + np.random.normal(0, 0.003, n))
    high_prices = np.maximum(open_prices, close_prices) * (1.0 + np.abs(np.random.normal(0, 0.005, n)))
    low_prices = np.minimum(open_prices, close_prices) * (1.0 - np.abs(np.random.normal(0, 0.005, n)))
    volumes = np.random.lognormal(mean=16.0, sigma=0.4, size=n)

    df = pd.DataFrame(
        {
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volumes,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def fetch_ohlcv_data(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    cache_dir: str = "data/raw",
    allow_synthetic_fallback: bool = True,
) -> pd.DataFrame:
    """Fetch historical OHLCV data with local caching and offline fallback.

    1. Checks if local cached CSV file exists.
    2. If not cached, attempts fetching via yf.Ticker.history() or yf.download().
    3. Caches downloaded data locally to avoid subsequent network calls.
    4. Falls back to realistic synthetic data if network/firewall blocks Yahoo Finance.

    Args:
        ticker: Asset symbol (e.g. 'SPY', 'AAPL', 'BTC-USD').
        start: Start date string 'YYYY-MM-DD'.
        end: End date string 'YYYY-MM-DD'.
        interval: Data interval (e.g. '1d', '1h', '5m').
        cache_dir: Directory where downloaded data is cached.
        allow_synthetic_fallback: Whether to generate synthetic data if network fails.

    Returns:
        Cleaned OHLCV DataFrame indexed by timestamp.
    """
    import os

    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = f"{ticker}_{start}_{end}.csv".replace(":", "-")
    cache_path = os.path.join(cache_dir, cache_filename)

    # 1. Check local cache
    if os.path.exists(cache_path):
        print(f"[*] Loading cached market data from: {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        col_mapping = {col: col.capitalize() for col in df.columns}
        df = df.rename(columns=col_mapping)
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if all(col in df.columns for col in required_cols):
            return df[required_cols].dropna()

    # 2. Attempt fetching from yfinance
    print(f"[*] Querying Yahoo Finance for {ticker} ({start} to {end})...")
    raw_df = pd.DataFrame()

    try:
        # Ticker.history is generally more resilient to header and proxy configurations
        t = yf.Ticker(ticker)
        raw_df = t.history(start=start, end=end, interval=interval)
    except Exception as err:
        print(f"[!] yf.Ticker.history failed ({err}). Trying yf.download...")

    if raw_df.empty:
        try:
            raw_df = yf.download(
                ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False
            )
        except Exception as err:
            print(f"[!] yf.download failed: {err}")

    # 3. Handle network failure or empty response
    if raw_df.empty:
        if allow_synthetic_fallback:
            print(f"[!] Network connection to Yahoo Finance was blocked or timed out.")
            print(f"[*] Generating realistic synthetic OHLCV dataset for {ticker} as fallback...")
            synth_df = generate_synthetic_ohlcv(ticker=ticker, start=start, end=end)
            synth_df.to_csv(cache_path)
            return synth_df
        else:
            raise ConnectionError(
                f"Failed to download data for '{ticker}' and synthetic fallback is disabled."
            )

    # Clean multi-index if returned
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)

    # Standardize column names
    col_mapping = {col: col.capitalize() for col in raw_df.columns}
    df = raw_df.rename(columns=col_mapping)
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' missing from downloaded data.")

    cleaned_df = df[required_cols].dropna()

    # Save to cache for future offline runs
    cleaned_df.to_csv(cache_path)
    print(f"[*] Cached {len(cleaned_df)} bars to {cache_path}")
    return cleaned_df


def build_feature_dataset(
    df: pd.DataFrame,
    horizon: int = 5,
    hurdle_rate: float = 0.0,
    vol_windows: List[int] = [10, 20],
    ewma_spans: List[int] = [10, 20],
    accel_lags: List[int] = [3, 5],
) -> pd.DataFrame:
    """Construct full feature matrix and forward-looking prediction targets.

    Features include:
    - Log returns (1-step and horizon-step)
    - High-Low spread & Close-Open return (intraday volatility proxies)
    - Rolling return volatilities across specified windows
    - Stationarized EWMA deviations across multiple spans
    - Price acceleration momentum across multiple lags
    - Volume percentage change

    Target:
    - forward_return: Fractional return over horizon T: (P_{t+T} - P_t) / P_t
    - target: Binary indicator (1 if forward_return > hurdle_rate else 0)

    Args:
        df: Input DataFrame with standard OHLCV columns.
        horizon: Prediction horizon T.
        hurdle_rate: Minimum excess return threshold tau for positive class label.
        vol_windows: List of window sizes for rolling volatility.
        ewma_spans: List of spans for EWMA deviation features.
        accel_lags: List of discrete lags for acceleration momentum features.

    Returns:
        Cleaned DataFrame containing all engineered features and target columns.
    """
    features = pd.DataFrame(index=df.index)

    # Base price series (Close)
    close = df["Close"]
    volume = df["Volume"]

    # 1. Log returns
    features["log_return_1"] = compute_log_returns(close)
    features["log_return_5"] = (np.log(close) - np.log(close.shift(5))) / 5.0

    # 2. Intraday dynamics
    features["hl_spread"] = (df["High"] - df["Low"]) / close
    features["co_return"] = (close - df["Open"]) / df["Open"]

    # 3. Rolling sample volatility
    for w in vol_windows:
        features[f"vol_{w}"] = compute_rolling_volatility(
            features["log_return_1"], window=w, annualize=False
        )
    if len(df) >= 150:
        features["vol_60"] = compute_rolling_volatility(
            features["log_return_1"], window=60, annualize=False
        )

    # 4. Stationarized EWMA deviations
    for span in ewma_spans:
        features[f"ewma_dev_{span}"] = compute_ewma_deviation(close, span=span)

    # 5. Discrete acceleration momentum
    for lag in accel_lags:
        features[f"accel_mom_{lag}"] = compute_acceleration_momentum(close, lag=lag)

    # 6. Volume change (log ratio)
    features["volume_log_change"] = np.log(volume + 1.0) - np.log(volume.shift(1) + 1.0)

    # Target: Forward return over horizon T
    forward_price = close.shift(-horizon)
    features["forward_return"] = (forward_price - close) / close
    features["target"] = (features["forward_return"] > hurdle_rate).astype(np.float32)

    # Clean rows with NaN values resulting from rolling windows and forward shifts
    cleaned_df = features.dropna().copy()
    return cleaned_df


# ==============================================================================
# 3. Time-Series Sequence Generation & PyTorch Dataset
# ==============================================================================

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset yielding sequential input windows, binary targets, and forward returns."""

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        forward_returns: np.ndarray,
    ):
        """
        Args:
            sequences: 3D float32 array of shape (num_samples, sequence_length, num_features).
            targets: 1D or 2D float32 array of binary classification labels.
            forward_returns: 1D or 2D float32 array of actual forward returns (for Sharpe loss).
        """
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32).view(-1, 1)
        self.forward_returns = torch.as_tensor(forward_returns, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx], self.forward_returns[idx]


def create_sliding_sequences(
    feature_matrix: np.ndarray,
    target_vector: np.ndarray,
    return_vector: np.ndarray,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert 2D tabular features into sliding temporal 3D sequences.

    For sequence ending at index t (features from t - sequence_length + 1 to t),
    the corresponding target is the forward return/label established at index t.

    Args:
        feature_matrix: 2D array of shape (T, num_features).
        target_vector: 1D array of shape (T,).
        return_vector: 1D array of shape (T,).
        sequence_length: Number of time steps per sequence window.

    Returns:
        sequences: Array of shape (T - sequence_length + 1, sequence_length, num_features).
        targets: Array of shape (T - sequence_length + 1,).
        returns: Array of shape (T - sequence_length + 1,).
    """
    num_samples = len(feature_matrix) - sequence_length + 1
    if num_samples <= 0:
        raise ValueError(
            f"Dataset length ({len(feature_matrix)}) is shorter than sequence_length ({sequence_length})."
        )

    sequences = np.empty((num_samples, sequence_length, feature_matrix.shape[1]), dtype=np.float32)
    targets = np.empty((num_samples,), dtype=np.float32)
    returns = np.empty((num_samples,), dtype=np.float32)

    for i in range(num_samples):
        sequences[i] = feature_matrix[i : i + sequence_length]
        targets[i] = target_vector[i + sequence_length - 1]
        returns[i] = return_vector[i + sequence_length - 1]

    return sequences, targets, returns


def prepare_dataloaders(
    df: pd.DataFrame,
    sequence_length: int = 30,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    batch_size: int = 64,
    scaler_type: str = "robust",
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, RobustScaler, List[str]]:
    """Chronologically split data, fit scaler on train set only, and build DataLoaders.

    Args:
        df: DataFrame containing feature columns, 'target', and 'forward_return'.
        sequence_length: Number of historical steps per model input sequence.
        train_ratio: Proportion of observations for training.
        val_ratio: Proportion of observations for validation.
        batch_size: Mini-batch size for training.
        scaler_type: 'robust' for RobustScaler, 'standard' for StandardScaler.
        num_workers: Number of DataLoader subprocesses.

    Returns:
        train_loader, val_loader, test_loader, fitted_scaler, feature_names.
    """
    feature_cols = [c for c in df.columns if c not in ["target", "forward_return"]]
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    if len(train_df) <= sequence_length or len(val_df) <= sequence_length:
        raise ValueError("Insufficient data points in split subsets for given sequence_length.")

    # Select and fit scaler strictly on training split
    scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols].values)
    val_scaled = scaler.transform(val_df[feature_cols].values)
    test_scaled = scaler.transform(test_df[feature_cols].values)

    # Generate sequential window tensors
    X_train, y_train, ret_train = create_sliding_sequences(
        train_scaled, train_df["target"].values, train_df["forward_return"].values, sequence_length
    )
    X_val, y_val, ret_val = create_sliding_sequences(
        val_scaled, val_df["target"].values, val_df["forward_return"].values, sequence_length
    )
    X_test, y_test, ret_test = create_sliding_sequences(
        test_scaled, test_df["target"].values, test_df["forward_return"].values, sequence_length
    )

    train_dataset = TimeSeriesDataset(X_train, y_train, ret_train)
    val_dataset = TimeSeriesDataset(X_val, y_val, ret_val)
    test_dataset = TimeSeriesDataset(X_test, y_test, ret_test)

    # Note: Sequential data can be shuffled across batches in training once sequences are formed,
    # or kept sequential for stateful regimes. By default, shuffle=True provides i.i.d. gradient steps.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, scaler, feature_cols

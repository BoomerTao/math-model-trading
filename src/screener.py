"""Stock screening and ranking engine using mathematical indicators and LSTM forecasting."""

from dataclasses import dataclass
import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch

from src.data_prep import (
    build_feature_dataset,
    fetch_ohlcv_data,
    create_sliding_sequences,
)
from src.model import LSTMForecaster


DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD", "MSFT", "AMZN", "META", "APLD"
]


@dataclass
class StockCandidate:
    ticker: str
    last_price: float
    pred_prob_up: float
    accel_momentum: float
    ewma_dev: float
    volatility: float
    composite_score: float
    rank: int = 0


class MarketScreener:
    """Scans an equity universe and scores candidates using LSTM return predictions and momentum acceleration."""

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        sequence_length: int = 30,
        model_dir: str = "models",
        device: Optional[torch.device] = None,
    ):
        self.universe = universe or DEFAULT_UNIVERSE
        self.sequence_length = sequence_length
        self.model_dir = model_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cached_models: Dict[str, LSTMForecaster] = {}

    def _load_or_create_model(self, ticker: str, num_features: int) -> LSTMForecaster:
        """Load trained model weights if present, or initialize an evaluation model."""
        checkpoint_path = os.path.join(self.model_dir, f"best_lstm_{ticker}_bce.pt")
        # Fallback to generic SPY model if ticker-specific model is not trained yet
        fallback_path = os.path.join(self.model_dir, "best_lstm_SPY_bce.pt")

        target_path = checkpoint_path if os.path.exists(checkpoint_path) else fallback_path

        model = LSTMForecaster(num_features=num_features, hidden_size=64, num_layers=2)
        if os.path.exists(target_path):
            checkpoint = torch.load(target_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
        else:
            # Fallback to evaluation mode
            model.eval()

        return model.to(self.device)

    def evaluate_ticker(self, ticker: str) -> Optional[StockCandidate]:
        """Fetch latest data, compute mathematical features, and generate forecast score."""
        try:
            # Fetch recent 2-year window for feature continuity
            df = fetch_ohlcv_data(
                ticker=ticker,
                start="2023-01-01",
                end=pd.Timestamp.now().strftime("%Y-%m-%d"),
                cache_dir="data/raw",
            )
            if len(df) < self.sequence_length + 30:
                print(f"[!] Insufficient history for {ticker} ({len(df)} bars). Skipping.")
                return None

            features_df = build_feature_dataset(df, horizon=5, hurdle_rate=0.0)
            feature_cols = [c for c in features_df.columns if c not in ["target", "forward_return"]]

            # Latest sequence
            recent_features = features_df[feature_cols].values[-self.sequence_length :]
            # Normalize with z-score on local window
            mean = np.mean(recent_features, axis=0)
            std = np.std(recent_features, axis=0) + 1e-6
            normalized_seq = (recent_features - mean) / std

            input_tensor = torch.tensor(
                normalized_seq.reshape(1, self.sequence_length, len(feature_cols)),
                dtype=torch.float32,
            ).to(self.device)

            model = self._load_or_create_model(ticker, len(feature_cols))
            with torch.no_grad():
                prob_up = float(model.predict_proba(input_tensor).item())

            last_row = features_df.iloc[-1]
            last_price = float(df["Close"].iloc[-1])
            accel_mom = float(last_row.get("accel_mom_5", 0.0))
            ewma_dev = float(last_row.get("ewma_dev_10", 0.0))
            vol = float(last_row.get("vol_20", 0.0))

            # Composite ranking score:
            # 60% probability of positive return, 25% price acceleration momentum, 15% EWMA pull
            # High acceleration with positive probability ranks highest
            accel_signal = np.clip(accel_mom * 100.0, -1.0, 1.0)
            composite = (0.60 * prob_up) + (0.25 * (accel_signal + 1.0) / 2.0) + (0.15 * max(0.0, ewma_dev))

            return StockCandidate(
                ticker=ticker,
                last_price=last_price,
                pred_prob_up=prob_up,
                accel_momentum=accel_mom,
                ewma_dev=ewma_dev,
                volatility=vol,
                composite_score=composite,
            )

        except Exception as e:
            print(f"[!] Error screening {ticker}: {e}")
            return None

    def scan_universe(self, top_n: int = 3) -> List[StockCandidate]:
        """Screen all universe assets and return top N ranked candidates."""
        candidates: List[StockCandidate] = []
        print(f"\n[*] Scanning universe ({len(self.universe)} tickers) for top momentum & return forecasts...")

        for ticker in self.universe:
            res = self.evaluate_ticker(ticker)
            if res is not None:
                candidates.append(res)

        # Sort descending by composite score
        candidates.sort(key=lambda c: c.composite_score, reverse=True)

        for i, cand in enumerate(candidates):
            cand.rank = i + 1

        return candidates[:top_n]

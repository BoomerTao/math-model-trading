from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_prep import (
    build_feature_dataset,
    fetch_ohlcv_data,
    create_sliding_sequences,
)
from src.model import LSTMForecaster


FALLBACK_UNIVERSE = [
    "SPY", "QQQ", "AAPL", "NVDA", "TSLA", "AMD", "MSFT", "AMZN", "META", "APLD"
]


def load_watchlist(config_path: str = "config/watchlist.json") -> Tuple[List[str], Dict[str, List[str]]]:
    """Load stock universe and sector mappings from JSON config file."""
    sectors_map = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
                sectors_map = data.get("sectors", {})
                tickers = data.get("tickers", [])
                if not tickers and sectors_map:
                    for sec_tickers in sectors_map.values():
                        tickers.extend(sec_tickers)
                if tickers:
                    seen = set()
                    unique = []
                    for t in tickers:
                        clean_t = str(t).upper().strip()
                        if clean_t not in seen:
                            seen.add(clean_t)
                            unique.append(clean_t)
                    return unique, sectors_map
        except Exception as e:
            print(f"[!] Warning: Failed to read {config_path} ({e}). Using default universe.")
    return FALLBACK_UNIVERSE, sectors_map


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
        if universe is None:
            loaded_tickers, _ = load_watchlist()
            self.universe = loaded_tickers
        else:
            self.universe = universe
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Market Momentum & LSTM Stock Screener")
    parser.add_argument("--list", action="store_true", help="List all stocks currently in the screening pool")
    parser.add_argument("--top-n", type=int, default=10, help="Number of ranked candidates to show (default: 10)")
    parser.add_argument("--config", type=str, default="config/watchlist.json", help="Path to watchlist config")
    args = parser.parse_args()

    tickers, sectors_map = load_watchlist(args.config)

    if args.list:
        print("\n" + "=" * 65)
        print(f"        AI & MOMENTUM SCREENING UNIVERSE ({len(tickers)} tickers)")
        print("=" * 65)
        if sectors_map:
            for sector, syms in sectors_map.items():
                print(f"\n[{sector}] ({len(syms)} stocks):")
                print("  " + ", ".join(syms))
        else:
            for i, ticker in enumerate(tickers, 1):
                print(f" {i:2d}. {ticker}")
        print("\n" + "=" * 65)
        print("To edit this pool, modify: config/watchlist.json\n")
    else:
        screener = MarketScreener(universe=tickers)
        top_picks = screener.scan_universe(top_n=args.top_n)

        print("\n" + "=" * 75)
        print("                      LIVE STOCK POOL RANKINGS")
        print("=" * 75)
        print(f"{'Rank':<6} {'Ticker':<7} {'Price':>9} {'P(Up)':>8} {'Acceleration':>15} {'EWMA Spread':>13} {'Score':>8}")
        print("-" * 75)
        for c in top_picks:
            print(
                f"#{c.rank:<5} {c.ticker:<7} ${c.last_price:>8.2f} "
                f"{c.pred_prob_up * 100:>7.1f}% {c.accel_momentum:>+15.5f} "
                f"{c.ewma_dev * 100:>+12.2f}% {c.composite_score:>8.3f}"
            )
        print("=" * 75 + "\n")


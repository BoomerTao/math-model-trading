"""Training, validation, and evaluation engine for quantitative LSTM forecasting models.

Supports:
- Optimization via BCEWithLogitsLoss or custom DifferentialSharpeLoss
- Adam optimizer with gradient clipping and learning rate scheduling
- Out-of-sample directional metrics (Accuracy, ROC-AUC, F1)
- Simulated strategy performance metrics (Cumulative Return, Realized Sharpe Ratio)
"""

import argparse
import os
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is on sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_prep import (
    build_feature_dataset,
    fetch_ohlcv_data,
    prepare_dataloaders,
)
from src.model import DifferentialSharpeLoss, LSTMForecaster


# ==============================================================================
# 1. Evaluation & Backtesting Utilities
# ==============================================================================

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    loss_type: str,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model performance on validation or test DataLoader.

    Computes:
    - Loss (BCE or Sharpe)
    - Classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC
    - Backtest proxy: Strategy Cumulative Return, Realized Sharpe Ratio

    Args:
        model: Trained or checkpointed PyTorch model.
        loader: DataLoader to evaluate.
        criterion: Loss criterion instance.
        loss_type: 'bce' or 'sharpe'.
        device: Target torch device.

    Returns:
        Dictionary of computed metric names and values.
    """
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    all_returns = []

    with torch.no_grad():
        for sequences, targets, returns in loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            returns = returns.to(device)

            logits = model(sequences)

            if loss_type == "sharpe":
                loss = criterion(logits, returns)
            else:
                loss = criterion(logits, targets)

            total_loss += loss.item() * sequences.size(0)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_returns.append(returns.cpu().numpy())

    n_samples = len(loader.dataset)
    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0

    logits_arr = np.vstack(all_logits).flatten()
    targets_arr = np.vstack(all_targets).flatten()
    returns_arr = np.vstack(all_returns).flatten()

    # Probabilities and discrete binary predictions
    probs_arr = 1.0 / (1.0 + np.exp(-logits_arr))
    preds_arr = (probs_arr >= 0.5).astype(int)

    # Classification metrics
    acc = accuracy_score(targets_arr, preds_arr)
    prec = precision_score(targets_arr, preds_arr, zero_division=0)
    rec = recall_score(targets_arr, preds_arr, zero_division=0)
    f1 = f1_score(targets_arr, preds_arr, zero_division=0)

    try:
        auc = roc_auc_score(targets_arr, probs_arr)
    except ValueError:
        # Handles edge cases where only a single class is present in batch
        auc = 0.5

    # Strategy simulation: Continuous position sizing s in [-1, 1] via tanh
    positions_arr = np.tanh(logits_arr)
    strategy_returns = positions_arr * returns_arr
    mean_strat = np.mean(strategy_returns)
    std_strat = np.std(strategy_returns) + 1e-8
    realized_sharpe = float(mean_strat / std_strat * np.sqrt(252))  # Annualized proxy
    cum_strat_return = float(np.prod(1.0 + strategy_returns) - 1.0)
    cum_benchmark_return = float(np.prod(1.0 + returns_arr) - 1.0)

    return {
        "loss": avg_loss,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "strategy_cum_return": cum_strat_return,
        "benchmark_cum_return": cum_benchmark_return,
        "realized_sharpe": realized_sharpe,
    }


# ==============================================================================
# 2. Main Training Loop
# ==============================================================================

def train_forecaster(
    ticker: str = "SPY",
    start_date: str = "2018-01-01",
    end_date: str = "2024-01-01",
    sequence_length: int = 30,
    horizon: int = 5,
    hurdle_rate: float = 0.0,
    hidden_size: int = 64,
    num_layers: int = 2,
    lstm_dropout: float = 0.2,
    fc_dropout: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 50,
    loss_type: str = "bce",
    patience: int = 10,
    checkpoint_dir: str = "models",
    device_name: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, float]]:
    """Execute end-to-end training and evaluation pipeline.

    Args:
        ticker: Asset symbol to fetch.
        start_date: Historical data start date.
        end_date: Historical data end date.
        sequence_length: Number of time steps in sequence window.
        horizon: Prediction horizon T.
        hurdle_rate: Target return hurdle tau.
        hidden_size: Hidden dimension of LSTM cells.
        num_layers: Number of LSTM layers.
        lstm_dropout: LSTM dropout rate.
        fc_dropout: Dense head dropout rate.
        lr: Initial learning rate.
        weight_decay: L2 regularization coefficient.
        batch_size: Mini-batch size.
        epochs: Maximum training epochs.
        loss_type: 'bce' for Binary Cross-Entropy, 'sharpe' for Differential Sharpe Loss.
        patience: Early stopping patience epochs.
        checkpoint_dir: Directory to save the best model checkpoint.
        device_name: Optional device string ('cpu', 'cuda').

    Returns:
        Trained model instance and final test set metrics dictionary.
    """
    device = torch.device(
        device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[*] Training Device: {device}")

    # 1. Ingestion & Feature Engineering
    print(f"[*] Fetching historical OHLCV data for {ticker} ({start_date} to {end_date})...")
    raw_df = fetch_ohlcv_data(ticker=ticker, start=start_date, end=end_date)
    print(f"[*] Downloaded {len(raw_df)} bars. Engineering features...")

    features_df = build_feature_dataset(
        raw_df, horizon=horizon, hurdle_rate=hurdle_rate
    )
    print(f"[*] Built feature matrix: {features_df.shape[0]} valid rows, {features_df.shape[1] - 2} features.")

    # 2. Dataloaders
    train_loader, val_loader, test_loader, scaler, feature_names = prepare_dataloaders(
        features_df,
        sequence_length=sequence_length,
        train_ratio=0.70,
        val_ratio=0.15,
        batch_size=batch_size,
    )
    print(f"[*] Features: {feature_names}")
    print(f"[*] Sequences generated — Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # 3. Model Architecture & Optimization
    model = LSTMForecaster(
        num_features=len(feature_names),
        hidden_size=hidden_size,
        num_layers=num_layers,
        lstm_dropout=lstm_dropout,
        fc_dropout=fc_dropout,
    ).to(device)

    if loss_type == "sharpe":
        criterion = DifferentialSharpeLoss(turnover_penalty=0.01)
        print("[*] Objective: Custom Differential Sharpe Ratio Loss")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("[*] Objective: Binary Cross-Entropy Loss (BCEWithLogitsLoss)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    # 4. Checkpoint Directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, f"best_lstm_{ticker}_{loss_type}.pt")

    # 5. Training Loop with Early Stopping
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print("\n--- Starting Training Loop ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for sequences, targets, returns in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            returns = returns.to(device)

            optimizer.zero_grad()
            logits = model(sequences)

            if loss_type == "sharpe":
                loss = criterion(logits, returns)
            else:
                loss = criterion(logits, targets)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * sequences.size(0)

        epoch_train_loss = train_loss / len(train_loader.dataset)
        val_metrics = evaluate_model(model, val_loader, criterion, loss_type, device)
        scheduler.step(val_metrics["loss"])

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Sharpe: {val_metrics['realized_sharpe']:.2f}"
        )

        # Early stopping logic
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler": scaler,
                    "features": feature_names,
                    "val_metrics": val_metrics,
                },
                best_checkpoint_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[!] Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # 6. Final Evaluation on Out-of-Sample Test Set
    print(f"\n[*] Loading best checkpoint from {best_checkpoint_path} for final test evaluation...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_model(model, test_loader, criterion, loss_type, device)
    print("\n================ OUT-OF-SAMPLE TEST RESULTS ================")
    print(f"Test Loss:             {test_metrics['loss']:.4f}")
    print(f"Test ROC-AUC:          {test_metrics['roc_auc']:.4f}")
    print(f"Test Accuracy:         {test_metrics['accuracy']:.4f}")
    print(f"Test Precision:        {test_metrics['precision']:.4f}")
    print(f"Test Recall:           {test_metrics['recall']:.4f}")
    print(f"Test F1 Score:         {test_metrics['f1']:.4f}")
    print(f"Realized Sharpe Ratio: {test_metrics['realized_sharpe']:.2f}")
    print(f"Strategy Cum Return:   {test_metrics['strategy_cum_return'] * 100:.2f}%")
    print(f"Benchmark Cum Return:  {test_metrics['benchmark_cum_return'] * 100:.2f}%")
    print("============================================================")

    return model, test_metrics


# ==============================================================================
# 3. CLI Entry Point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Asset Return Forecaster")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--start", type=str, default="2018-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--seq-len", type=int, default=30, help="Sequence window length (default: 30)")
    parser.add_argument("--horizon", type=int, default=5, help="Forward return horizon T (default: 5)")
    parser.add_argument("--hurdle", type=float, default=0.0, help="Forward return hurdle rate tau (default: 0.0)")
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden units (default: 64)")
    parser.add_argument("--num-layers", type=int, default=2, help="LSTM depth (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--epochs", type=int, default=30, help="Max epochs (default: 30)")
    parser.add_argument("--loss-type", type=str, choices=["bce", "sharpe"], default="bce", help="Loss function")
    parser.add_argument("--device", type=str, default=None, help="Target device ('cpu' or 'cuda')")

    args = parser.parse_args()

    train_forecaster(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        sequence_length=args.seq_len,
        horizon=args.horizon,
        hurdle_rate=args.hurdle,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        loss_type=args.loss_type,
        device_name=args.device,
    )

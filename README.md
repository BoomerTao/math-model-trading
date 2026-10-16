# Mathematical Model Trading (Quantitative LSTM Forecaster)

A quantitative finance forecasting framework in Python using PyTorch and mathematical indicators to predict asset returns.

## Architecture & Mathematical Indicators

- **Logarithmic Returns:** Multi-step continuous compounding $r_t = \ln(P_t) - \ln(P_{t-1})$.
- **Rolling Volatility:** Rolling sample standard deviation $\sigma_{t, w}$ to capture variance clustering.
- **Stationarized EWMA Spread:** Relative price deviation $\delta_{\text{EWMA}, t} = \frac{P_t - \text{EWMA}_t}{\text{EWMA}_t}$ ensuring stationary bounded inputs.
- **Acceleration Momentum Factor:** Discrete second derivative of log price $a_t = \frac{\Delta^2 \ln(P_t)}{k^2}$ to identify inflection points and momentum exhaustion.
- **LSTM Forecaster:** Deep 2-layer recurrent network consuming 3D tensors `[batch_size, sequence_length, num_features]`.
- **Loss Functions:**
  - Standard Binary Cross-Entropy (`BCEWithLogitsLoss`) for directional classification over horizon $T$ against hurdle rate $\tau$.
  - Custom `DifferentialSharpeLoss` mapping logits to positions $s_t = \tanh(z_t)$ to directly optimize the portfolio Sharpe Ratio.

---

## Project Structure

```text
math-model-trading/
├── data/                  # Local cache for OHLCV data
├── notebooks/             # Exploratory analysis & feature visualizations
├── src/
│   ├── __init__.py
│   ├── data_prep.py       # Indicator math, yfinance ingestion, time-series sequence generation
│   ├── model.py           # PyTorch LSTMForecaster & DifferentialSharpeLoss
│   └── train.py           # Training loop, early stopping, evaluation & backtest metrics
├── tests/
│   └── test_pipeline.py   # Unit tests for mathematical consistency & tensor shapes
├── requirements.txt       # Dependencies
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Train with Binary Cross-Entropy (BCE)
```bash
python src/train.py --ticker SPY --start 2018-01-01 --end 2024-01-01 --seq-len 30 --horizon 5 --loss-type bce --epochs 30
```

### 2. Train with Custom Differential Sharpe Ratio Loss
```bash
python src/train.py --ticker SPY --start 2018-01-01 --end 2024-01-01 --seq-len 30 --horizon 5 --loss-type sharpe --epochs 30
```

### 3. Run Automated Paper Trading Bot

#### Run Once (Check Market & Execute If Open)
```bash
python src/bot.py --top-n 3
```

#### Run in Continuous Daemon Mode (Sleeps until Market Open, then polls every 15m)
```bash
python src/bot.py --daemon --poll-interval-minutes 15 --top-n 3
```

#### Test Mode (Simulate execution immediately after hours)
```bash
python src/bot.py --allow-after-hours --top-n 3
```

---

## Unit Tests
```bash
python -m pytest tests/
```

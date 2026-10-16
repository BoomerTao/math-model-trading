"""Unit tests for mathematical indicators, model dimensions, and loss functions."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_prep import (
    build_feature_dataset,
    compute_acceleration_momentum,
    compute_ewma_deviation,
    compute_log_returns,
    compute_rolling_volatility,
    create_sliding_sequences,
)
from src.model import DifferentialSharpeLoss, LSTMForecaster


def test_log_returns():
    prices = pd.Series([100.0, 110.0, 105.0])
    log_rets = compute_log_returns(prices)
    assert np.isnan(log_rets.iloc[0])
    assert np.isclose(log_rets.iloc[1], np.log(110.0 / 100.0))
    assert np.isclose(log_rets.iloc[2], np.log(105.0 / 110.0))


def test_rolling_volatility():
    # Constant returns should yield zero volatility
    log_rets = pd.Series([0.01] * 25)
    vol = compute_rolling_volatility(log_rets, window=20, annualize=False)
    assert np.isclose(vol.iloc[19], 0.0, atol=1e-8)


def test_ewma_deviation():
    # If prices are flat, deviation from EWMA is 0
    prices = pd.Series([100.0] * 30)
    dev = compute_ewma_deviation(prices, span=10)
    assert np.isclose(dev.iloc[-1], 0.0, atol=1e-8)


def test_acceleration_momentum():
    # Constant price sequence should yield 0 acceleration
    prices = pd.Series([100.0] * 30)
    accel = compute_acceleration_momentum(prices, lag=5, smooth_span=None)
    assert np.isclose(accel.iloc[-1], 0.0, atol=1e-8)


def test_create_sliding_sequences():
    features = np.ones((50, 6), dtype=np.float32)
    targets = np.ones((50,), dtype=np.float32)
    returns = np.ones((50,), dtype=np.float32) * 0.02
    seq_len = 10

    seqs, tgts, rets = create_sliding_sequences(features, targets, returns, sequence_length=seq_len)
    assert seqs.shape == (41, 10, 6)
    assert tgts.shape == (41,)
    assert rets.shape == (41,)


def test_lstm_forecaster_dimensions():
    batch_size = 16
    seq_len = 20
    num_features = 8

    model = LSTMForecaster(num_features=num_features, hidden_size=32, num_layers=2)
    dummy_input = torch.randn(batch_size, seq_len, num_features)

    logits = model(dummy_input)
    assert logits.shape == (batch_size, 1)

    probs = model.predict_proba(dummy_input)
    assert probs.shape == (batch_size, 1)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

    positions = model.predict_position(dummy_input)
    assert positions.shape == (batch_size, 1)
    assert torch.all(positions >= -1.0) and torch.all(positions <= 1.0)


def test_differential_sharpe_loss():
    loss_fn = DifferentialSharpeLoss()
    logits = torch.tensor([[1.5], [2.0], [0.8], [1.2]], requires_grad=True)
    # Positive forward returns matching positive positions
    forward_returns = torch.tensor([[0.05], [0.04], [0.02], [0.03]])

    loss = loss_fn(logits, forward_returns)
    assert loss.dim() == 0  # Scalar
    assert loss.item() < 0  # Negative loss represents positive Sharpe ratio

    # Test backward pass
    loss.backward()
    assert logits.grad is not None

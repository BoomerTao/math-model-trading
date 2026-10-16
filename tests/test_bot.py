"""Unit tests for Paper Broker and Market Screener."""

import os
import pytest
from src.paper_broker import PaperBroker


def test_paper_broker_initialization():
    broker = PaperBroker(initial_cash=50_000.0, state_file="data/test_portfolio.json")
    assert broker.cash == 50_000.0
    assert len(broker.positions) == 0


def test_paper_broker_buy_sell_flow():
    test_state_file = "data/test_portfolio_flow.json"
    if os.path.exists(test_state_file):
        os.remove(test_state_file)

    broker = PaperBroker(initial_cash=100_000.0, state_file=test_state_file)

    # Buy AAPL with $10,000 at $150
    success = broker.buy("AAPL", dollar_amount=10_000.0, current_price=150.0, reason="Test Buy")
    assert success is True
    assert "AAPL" in broker.positions
    assert broker.cash < 100_000.0

    # Sell all AAPL at $160 (profitable)
    success_sell = broker.sell("AAPL", current_price=160.0, reason="Test Sell")
    assert success_sell is True
    assert "AAPL" not in broker.positions
    assert broker.cash > 100_000.0  # Profited after fees

    summary = broker.get_portfolio_summary({})
    assert summary["total_pnl"] > 0

    if os.path.exists(test_state_file):
        os.remove(test_state_file)

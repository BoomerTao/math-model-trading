"""Simulated Paper Brokerage & Portfolio Management Engine.

Maintains cash, open positions, unrealized/realized PnL, slippage modeling,
and persistent ledger tracking to JSON/CSV without requiring paid API keys.
"""

from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional


class PaperBroker:
    """Manages simulated trading accounts, order execution, position tracking, and PnL reporting."""

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        slippage_bps: float = 5.0,  # 5 basis points = 0.05%
        state_file: str = "data/paper_portfolio.json",
    ):
        self.initial_cash = initial_cash
        self.slippage_rate = slippage_bps / 10_000.0
        self.state_file = state_file

        self.cash: float = initial_cash
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []

        self.load_state()

    def load_state(self) -> None:
        """Load portfolio state from JSON if it exists."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.cash = data.get("cash", self.initial_cash)
                    self.positions = data.get("positions", {})
                    self.trade_history = data.get("trade_history", [])
                    print(f"[*] Loaded existing paper portfolio from {self.state_file}")
            except Exception as e:
                print(f"[!] Warning: Could not parse {self.state_file} ({e}). Starting fresh.")

    def save_state(self) -> None:
        """Persist current portfolio state to disk."""
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        data = {
            "initial_cash": self.initial_cash,
            "cash": round(self.cash, 2),
            "positions": self.positions,
            "trade_history": self.trade_history,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

    def buy(
        self,
        ticker: str,
        dollar_amount: float,
        current_price: float,
        reason: str = "",
    ) -> bool:
        """Execute a simulated market BUY order.

        Args:
            ticker: Asset symbol.
            dollar_amount: Total capital to commit.
            current_price: Current market price.
            reason: Strategy rationale for the order.

        Returns:
            True if order filled, False if insufficient funds.
        """
        if dollar_amount <= 0 or current_price <= 0:
            return False

        if dollar_amount > self.cash:
            dollar_amount = self.cash

        if dollar_amount < 50.0:  # Minimum order threshold
            return False

        # Apply slippage: buy fills slightly higher than mid price
        exec_price = current_price * (1.0 + self.slippage_rate)
        shares_to_buy = dollar_amount / exec_price
        cost = shares_to_buy * exec_price

        self.cash -= cost

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        if ticker in self.positions:
            existing = self.positions[ticker]
            new_qty = existing["qty"] + shares_to_buy
            avg_price = ((existing["qty"] * existing["avg_price"]) + cost) / new_qty
            self.positions[ticker]["qty"] = new_qty
            self.positions[ticker]["avg_price"] = avg_price
        else:
            self.positions[ticker] = {
                "qty": shares_to_buy,
                "avg_price": exec_price,
                "entry_time": now_str,
            }

        trade_record = {
            "timestamp": now_str,
            "action": "BUY",
            "ticker": ticker,
            "shares": round(shares_to_buy, 4),
            "exec_price": round(exec_price, 2),
            "total_value": round(cost, 2),
            "reason": reason,
        }
        self.trade_history.append(trade_record)
        self.save_state()

        print(
            f" [EXECUTED BUY] {ticker}: {shares_to_buy:.2f} shares @ ${exec_price:.2f} "
            f"(${cost:,.2f}) | Cash Left: ${self.cash:,.2f}"
        )
        return True

    def sell(
        self,
        ticker: str,
        current_price: float,
        qty: Optional[float] = None,
        reason: str = "",
    ) -> bool:
        """Execute a simulated market SELL order.

        Args:
            ticker: Asset symbol.
            current_price: Current market price.
            qty: Quantity to sell (None to liquidate entire position).
            reason: Strategy rationale for closing/reducing.

        Returns:
            True if executed, False if position not found.
        """
        if ticker not in self.positions or current_price <= 0:
            return False

        pos = self.positions[ticker]
        shares_to_sell = pos["qty"] if qty is None or qty >= pos["qty"] else qty

        # Apply slippage: sell fills slightly lower
        exec_price = current_price * (1.0 - self.slippage_rate)
        proceeds = shares_to_sell * exec_price
        cost_basis = shares_to_sell * pos["avg_price"]
        realized_pnl = proceeds - cost_basis

        self.cash += proceeds

        if shares_to_sell >= pos["qty"]:
            del self.positions[ticker]
        else:
            self.positions[ticker]["qty"] -= shares_to_sell

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        trade_record = {
            "timestamp": now_str,
            "action": "SELL",
            "ticker": ticker,
            "shares": round(shares_to_sell, 4),
            "exec_price": round(exec_price, 2),
            "total_value": round(proceeds, 2),
            "realized_pnl": round(realized_pnl, 2),
            "reason": reason,
        }
        self.trade_history.append(trade_record)
        self.save_state()

        pnl_symbol = "+" if realized_pnl >= 0 else ""
        print(
            f" [EXECUTED SELL] {ticker}: {shares_to_sell:.2f} shares @ ${exec_price:.2f} "
            f"| Realized PnL: {pnl_symbol}${realized_pnl:,.2f} | Reason: {reason}"
        )
        return True

    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Compute portfolio equity, unrealized PnL, and allocation weights."""
        holdings_value = 0.0
        position_details = []

        for ticker, pos in self.positions.items():
            curr_px = current_prices.get(ticker, pos["avg_price"])
            pos_val = pos["qty"] * curr_px
            cost_basis = pos["qty"] * pos["avg_price"]
            unrealized = pos_val - cost_basis
            unrealized_pct = (unrealized / cost_basis * 100.0) if cost_basis > 0 else 0.0

            holdings_value += pos_val
            position_details.append(
                {
                    "ticker": ticker,
                    "shares": round(pos["qty"], 4),
                    "avg_price": round(pos["avg_price"], 2),
                    "current_price": round(curr_px, 2),
                    "market_value": round(pos_val, 2),
                    "unrealized_pnl": round(unrealized, 2),
                    "unrealized_pct": round(unrealized_pct, 2),
                }
            )

        total_equity = self.cash + holdings_value
        total_pnl = total_equity - self.initial_cash
        total_pnl_pct = (total_pnl / self.initial_cash) * 100.0

        return {
            "initial_cash": self.initial_cash,
            "cash": round(self.cash, 2),
            "holdings_value": round(holdings_value, 2),
            "total_equity": round(total_equity, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "num_positions": len(self.positions),
            "positions": position_details,
        }


if __name__ == "__main__":
    broker = PaperBroker()
    summary = broker.get_portfolio_summary({})

    print("\n" + "=" * 80)
    print("                    PAPER TRADING ACCOUNT SUMMARY")
    print("=" * 80)
    pnl_sign = "+" if summary["total_pnl"] >= 0 else ""
    print(f"Total Equity:      ${summary['total_equity']:>12,.2f}")
    print(f"Available Cash:    ${summary['cash']:>12,.2f}")
    print(f"Holdings Value:    ${summary['holdings_value']:>12,.2f}")
    print(f"Total PnL:         {pnl_sign}${summary['total_pnl']:>11,.2f} ({pnl_sign}{summary['total_pnl_pct']:.2f}%)")
    print(f"Active Positions:  {summary['num_positions']}")
    print("-" * 80)

    print("\n[ACTIVE OPEN POSITIONS]")
    if summary["positions"]:
        print(f"{'Ticker':<7} {'Shares':>9} {'Entry Price':>13} {'Cost Basis':>13} {'Entry Time'}")
        print("-" * 80)
        for p in summary["positions"]:
            pos_info = broker.positions.get(p["ticker"], {})
            entry_t = pos_info.get("entry_time", "N/A")
            print(f"{p['ticker']:<7} {p['shares']:>9.2f} ${p['avg_price']:>12.2f} ${p['market_value']:>12,.2f}  {entry_t}")
    else:
        print("  No open positions (100% Cash).")

    print("\n" + "=" * 80)
    print("                    EXECUTED TRADE HISTORY LEDGER")
    print("=" * 80)
    if broker.trade_history:
        print(f"{'Timestamp':<24} {'Action':<6} {'Ticker':<7} {'Shares':>9} {'Price':>9} {'Total':>11} {'Reason'}")
        print("-" * 80)
        for t in broker.trade_history:
            print(
                f"{t['timestamp']:<24} {t['action']:<6} {t['ticker']:<7} "
                f"{t['shares']:>9.2f} ${t['exec_price']:>8.2f} ${t['total_value']:>10,.2f}  {t.get('reason', '')}"
            )
    else:
        print("  No trades executed yet.")
    print("=" * 80 + "\n")


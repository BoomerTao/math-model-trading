"""Automated Paper Trading Bot with Real Market Schedule Awareness.

Scans the universe for top-ranked momentum & return candidates using the LSTM model,
and automatically manages a simulated paper portfolio when the market is open.
"""

import argparse
from datetime import datetime, time as dtime, timedelta
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple
import pytz

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.paper_broker import PaperBroker
from src.screener import MarketScreener, StockCandidate


# ==============================================================================
# 1. Market Hours & Schedule Management
# ==============================================================================

def get_market_status() -> Tuple[bool, str, Optional[timedelta]]:
    """Determine if the US Equity Market (NYSE/NASDAQ) is currently open.

    Regular Trading Hours: Mon-Fri 09:30 AM - 04:00 PM US Eastern Time.

    Returns:
        (is_open, status_message, time_until_open)
    """
    eastern = pytz.timezone("US/Eastern")
    now_et = datetime.now(eastern)

    # Weekday check: Monday=0, Sunday=6
    weekday = now_et.weekday()
    market_open_time = dtime(9, 30)
    market_close_time = dtime(16, 0)

    if weekday >= 5:  # Saturday or Sunday
        days_ahead = 7 - weekday  # Next Monday
        next_open = (now_et + timedelta(days=days_ahead)).replace(
            hour=9, minute=30, second=0, microsecond=0
        )
        time_to_open = next_open - now_et
        return False, f"Weekend (Market Closed). Next open on Monday at 09:30 AM ET.", time_to_open

    current_time = now_et.time()

    if current_time < market_open_time:
        next_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        time_to_open = next_open - now_et
        return False, f"Pre-Market (Closed). Opens at 09:30 AM ET.", time_to_open

    if current_time > market_close_time:
        next_day = now_et + timedelta(days=1 if weekday < 4 else 3)
        next_open = next_day.replace(hour=9, minute=30, second=0, microsecond=0)
        time_to_open = next_open - now_et
        return False, f"Post-Market (Closed). Opens tomorrow at 09:30 AM ET.", time_to_open

    time_to_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0) - now_et
    return True, f"Regular Trading Session OPEN. Closes in {str(time_to_close).split('.')[0]}", None


# ==============================================================================
# 2. Automated Paper Trading Bot
# ==============================================================================

class PaperTradingBot:
    """Orchestrates screening, risk management, and order execution for paper accounts."""

    def __init__(
        self,
        top_n: int = 3,
        initial_capital: float = 100_000.0,
        stop_loss_pct: float = 0.05,     # 5% max drawdown per position
        take_profit_pct: float = 0.15,   # 15% profit target
        allow_after_hours: bool = False,
    ):
        self.top_n = top_n
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.allow_after_hours = allow_after_hours

        self.broker = PaperBroker(initial_cash=initial_capital)
        self.screener = MarketScreener()

    def run_cycle(self) -> None:
        """Execute one complete scan, risk check, and rebalance cycle."""
        is_open, status_msg, time_to_open = get_market_status()

        print("\n" + "=" * 65)
        print(f"[*] US MARKET STATUS: {status_msg}")
        print("=" * 65)

        if not is_open and not self.allow_after_hours:
            print(f"[!] Real market is currently closed. Time until open: {str(time_to_open).split('.')[0]}")
            print("[*] To force an immediate test trade right now, run with: --allow-after-hours")
            return

        # 1. Screen Universe
        top_candidates = self.screener.scan_universe(top_n=self.top_n)
        if not top_candidates:
            print("[!] Screener yielded no qualified candidates.")
            return

        print("\n--- TOP RANKED ASSETS IDENTIFIED BY LSTM & MOMENTUM ---")
        candidate_dict: Dict[str, StockCandidate] = {}
        current_prices: Dict[str, float] = {}

        for cand in top_candidates:
            candidate_dict[cand.ticker] = cand
            current_prices[cand.ticker] = cand.last_price
            print(
                f"Rank #{cand.rank} | {cand.ticker:<6} | Price: ${cand.last_price:>7.2f} | "
                f"P(Up): {cand.pred_prob_up * 100:>5.1f}% | Accel: {cand.accel_momentum:>+.5f} | "
                f"Score: {cand.composite_score:.3f}"
            )

        # 2. Check Existing Positions for Stop-Loss / Take-Profit or Rank Drop
        summary = self.broker.get_portfolio_summary(current_prices)
        for pos in summary["positions"]:
            ticker = pos["ticker"]
            curr_px = current_prices.get(ticker, pos["avg_price"])
            unrealized_pct = pos["unrealized_pct"] / 100.0

            # Stop-loss triggered
            if unrealized_pct <= -self.stop_loss_pct:
                self.broker.sell(
                    ticker=ticker,
                    current_price=curr_px,
                    reason=f"Stop-loss hit ({pos['unrealized_pct']:.1f}% <= -{self.stop_loss_pct * 100:.0f}%)",
                )
            # Take-profit triggered
            elif unrealized_pct >= self.take_profit_pct:
                self.broker.sell(
                    ticker=ticker,
                    current_price=curr_px,
                    reason=f"Take-profit hit (+{pos['unrealized_pct']:.1f}% >= +{self.take_profit_pct * 100:.0f}%)",
                )
            # Dropped out of top picks
            elif ticker not in candidate_dict:
                self.broker.sell(
                    ticker=ticker,
                    current_price=curr_px,
                    reason=f"Rotated out of top {self.top_n} ranked momentum stocks",
                )

        # 3. Allocate Capital to Top Picks
        summary = self.broker.get_portfolio_summary(current_prices)
        total_equity = summary["total_equity"]
        target_per_asset = total_equity / self.top_n

        for cand in top_candidates:
            ticker = cand.ticker
            price = cand.last_price

            existing_pos = self.broker.positions.get(ticker)
            current_val = (existing_pos["qty"] * price) if existing_pos else 0.0
            deficit = target_per_asset - current_val

            # Buy if under-allocated by at least $500 and cash is available
            if deficit > 500.0 and self.broker.cash >= 100.0:
                alloc = min(deficit, self.broker.cash)
                self.broker.buy(
                    ticker=ticker,
                    dollar_amount=alloc,
                    current_price=price,
                    reason=f"Top #{cand.rank} rank allocation (P(Up)={cand.pred_prob_up * 100:.1f}%)",
                )

        # 4. Display Updated Portfolio Dashboard
        self.print_dashboard(current_prices)

    def print_dashboard(self, current_prices: Dict[str, float]) -> None:
        """Display real-time paper portfolio metrics."""
        summary = self.broker.get_portfolio_summary(current_prices)

        print("\n" + "=" * 65)
        print("                  PAPER TRADING PORTFOLIO DASHBOARD             ")
        print("=" * 65)
        pnl_symbol = "+" if summary["total_pnl"] >= 0 else ""
        print(f"Total Equity:    ${summary['total_equity']:>12,.2f}")
        print(f"Available Cash:  ${summary['cash']:>12,.2f}")
        print(f"Holdings Value:  ${summary['holdings_value']:>12,.2f}")
        print(f"Total PnL:       {pnl_symbol}${summary['total_pnl']:>11,.2f} ({pnl_symbol}{summary['total_pnl_pct']:.2f}%)")
        print(f"Open Positions:  {summary['num_positions']}")
        print("-" * 65)

        if summary["positions"]:
            print(f"{'Ticker':<7} {'Shares':>9} {'Avg Price':>11} {'Current':>10} {'Value':>11} {'PnL ($)':>10} {'PnL (%)':>8}")
            print("-" * 65)
            for p in summary["positions"]:
                pnl_str = f"{'+' if p['unrealized_pnl'] >= 0 else ''}${p['unrealized_pnl']:,.2f}"
                pct_str = f"{'+' if p['unrealized_pct'] >= 0 else ''}{p['unrealized_pct']:.1f}%"
                print(
                    f"{p['ticker']:<7} {p['shares']:>9.2f} ${p['avg_price']:>10.2f} "
                    f"${p['current_price']:>9.2f} ${p['market_value']:>10,.2f} {pnl_str:>10} {pct_str:>8}"
                )
        else:
            print("No open positions. 100% Cash.")
        print("=" * 65 + "\n")


# ==============================================================================
# 3. CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Automated Paper Trading Bot")
    parser.add_argument("--top-n", type=int, default=3, help="Number of top assets to hold (default: 3)")
    parser.add_argument("--capital", type=float, default=100_000.0, help="Initial paper capital ($)")
    parser.add_argument("--stop-loss", type=float, default=0.05, help="Stop loss fraction (default: 0.05 = 5%)")
    parser.add_argument("--take-profit", type=float, default=0.15, help="Take profit fraction (default: 0.15 = 15%)")
    parser.add_argument(
        "--allow-after-hours",
        action="store_true",
        help="Simulate execution immediately without waiting for real market open",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run continuously in background, sleeping until market open and polling periodically",
    )
    parser.add_argument(
        "--poll-interval-minutes",
        type=int,
        default=15,
        help="Minutes between re-scan cycles during market hours (default: 15)",
    )

    args = parser.parse_args()

    bot = PaperTradingBot(
        top_n=args.top_n,
        initial_capital=args.capital,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        allow_after_hours=args.allow_after_hours,
    )

    if not args.daemon:
        # Run single cycle
        bot.run_cycle()
    else:
        print("[*] Starting Bot in Daemon Mode (Continuous monitoring)...")
        while True:
            try:
                bot.run_cycle()
                is_open, _, time_to_open = get_market_status()

                if not is_open and not args.allow_after_hours:
                    # Sleep until market open, checking every 60s
                    sleep_secs = min(300, int(time_to_open.total_seconds())) if time_to_open else 300
                    print(f"[*] Sleeping {sleep_secs // 60} minutes until market opens...")
                    time.sleep(sleep_secs)
                else:
                    print(f"[*] Market open. Waiting {args.poll_interval_minutes} minutes until next scan...")
                    time.sleep(args.poll_interval_minutes * 60)

            except KeyboardInterrupt:
                print("\n[!] Bot stopped by user.")
                break
            except Exception as e:
                print(f"[!] Error in bot loop: {e}. Retrying in 60s...")
                time.sleep(60)


if __name__ == "__main__":
    main()

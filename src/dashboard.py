"""Interactive Intraday Equity Curve & Portfolio Dashboard.

Generates an interactive, responsive HTML visualizer for intraday portfolio equity,
trade markers, open position distribution, and PnL metrics.
Can open directly in the browser and auto-refresh.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import webbrowser

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.paper_broker import PaperBroker


def generate_dashboard_html(portfolio_file: str = "data/paper_portfolio.json") -> str:
    """Generate self-contained interactive HTML dashboard with Chart.js and Tailwind."""
    broker = PaperBroker(state_file=portfolio_file)
    summary = broker.get_portfolio_summary({})

    trades = broker.trade_history
    initial_cash = broker.initial_cash

    # Reconstruct intraday equity checkpoints from trade history
    time_points = ["09:30 AM (Open)"]
    equity_values = [initial_cash]
    trade_annotations = ["Initial Balance: $100,000"]

    running_equity = initial_cash
    for t in trades:
        time_str = t["timestamp"].split(" ")[1][:5] + " UTC"
        action = t["action"]
        ticker = t["ticker"]
        val = t["total_value"]

        # Calculate equity checkpoint
        if "realized_pnl" in t:
            running_equity += t["realized_pnl"]
        elif action == "BUY":
            # 5 bps slippage deduction on order
            running_equity -= (val * 0.0005)

        time_points.append(f"{time_str} ({action} {ticker})")
        equity_values.append(round(running_equity, 2))
        trade_annotations.append(f"{action} {ticker}: ${val:,.2f} ({t.get('reason', '')})")

    # Current checkpoint
    time_points.append("Current Live")
    equity_values.append(summary["total_equity"])
    trade_annotations.append(f"Active Equity: ${summary['total_equity']:,.2f}")

    pnl_sign = "+" if summary["total_pnl"] >= 0 else ""
    pnl_color = "#10b981" if summary["total_pnl"] >= 0 else "#ef4444"

    positions_json = json.dumps(summary["positions"])
    trade_history_json = json.dumps(broker.trade_history)
    time_points_json = json.dumps(time_points)
    equity_values_json = json.dumps(equity_values)
    annotations_json = json.dumps(trade_annotations)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Trading Bot — Intraday Equity Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ background-color: #0f172a; color: #f8fafc; font-family: ui-sans-serif, system-ui, sans-serif; }}
    .card {{ background-color: #1e293b; border: 1px solid #334155; }}
  </style>
</head>
<body class="p-4 md:p-8">
  <div class="max-w-6xl mx-auto space-y-6">
    
    <!-- Top Header & Metrics -->
    <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 card p-6 rounded-2xl shadow-xl">
      <div>
        <div class="flex items-center gap-3">
          <span class="inline-block w-3 h-3 rounded-full bg-emerald-400 animate-ping"></span>
          <h1 class="text-2xl font-bold tracking-tight">AI Paper Trading Dashboard</h1>
        </div>
        <p class="text-slate-400 text-sm mt-1">Autonomous Momentum & LSTM Portfolio Forecaster</p>
      </div>
      <div class="flex flex-wrap items-center gap-6">
        <div>
          <p class="text-xs text-slate-400 uppercase tracking-wider">Total Equity</p>
          <p class="text-3xl font-extrabold text-white">${summary['total_equity']:,.2f}</p>
        </div>
        <div class="border-l border-slate-700 pl-6">
          <p class="text-xs text-slate-400 uppercase tracking-wider">Total Return</p>
          <p class="text-3xl font-extrabold" style="color: {pnl_color};">{pnl_sign}${summary['total_pnl']:,.2f} ({pnl_sign}{summary['total_pnl_pct']:.2f}%)</p>
        </div>
        <div class="border-l border-slate-700 pl-6">
          <p class="text-xs text-slate-400 uppercase tracking-wider">Active Positions</p>
          <p class="text-3xl font-extrabold text-cyan-400">{summary['num_positions']}</p>
        </div>
      </div>
    </div>

    <!-- Main Chart Section -->
    <div class="card p-6 rounded-2xl shadow-xl space-y-4">
      <div class="flex justify-between items-center">
        <div>
          <h2 class="text-lg font-semibold text-slate-200">Intraday Equity Curve ($)</h2>
          <p class="text-xs text-slate-400">Mark-to-market portfolio value trajectory across trade executions</p>
        </div>
        <div class="flex gap-2">
          <span class="text-xs bg-slate-800 border border-slate-700 px-3 py-1.5 rounded-lg text-slate-300">Intraday</span>
        </div>
      </div>
      <div class="h-80 w-full">
        <canvas id="equityChart"></canvas>
      </div>
    </div>

    <!-- Active Positions & Allocation -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      
      <!-- Position Cards (2 cols) -->
      <div class="lg:col-span-2 card p-6 rounded-2xl shadow-xl space-y-4">
        <h2 class="text-lg font-semibold text-slate-200">Active AI Holdings</h2>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-sm">
            <thead>
              <tr class="text-xs text-slate-400 border-b border-slate-700">
                <th class="pb-3">Asset</th>
                <th class="pb-3 text-right">Shares</th>
                <th class="pb-3 text-right">Avg Price</th>
                <th class="pb-3 text-right">Allocation</th>
                <th class="pb-3 text-right">Entry Time</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-800" id="positionsTable">
            </tbody>
          </table>
        </div>
      </div>

      <!-- Donut Allocation Chart -->
      <div class="card p-6 rounded-2xl shadow-xl space-y-4 flex flex-col justify-between">
        <h2 class="text-lg font-semibold text-slate-200">Capital Distribution</h2>
        <div class="h-60 flex items-center justify-center">
          <canvas id="allocationChart"></canvas>
        </div>
        <p class="text-xs text-center text-slate-400">Equal-weight momentum allocation across top 3 assets</p>
      </div>

    </div>

    <!-- Trade Ledger Table -->
    <div class="card p-6 rounded-2xl shadow-xl space-y-4">
      <h2 class="text-lg font-semibold text-slate-200">Today's Executed Orders</h2>
      <div class="overflow-x-auto">
        <table class="w-full text-left text-sm">
          <thead>
            <tr class="text-xs text-slate-400 border-b border-slate-700">
              <th class="pb-3">Time</th>
              <th class="pb-3">Action</th>
              <th class="pb-3">Ticker</th>
              <th class="pb-3 text-right">Shares</th>
              <th class="pb-3 text-right">Fill Price</th>
              <th class="pb-3 text-right">Order Value</th>
              <th class="pb-3">Strategy Rationale</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-800" id="tradesTable">
          </tbody>
        </table>
      </div>
    </div>

  </div>

  <script>
    const timePoints = {time_points_json};
    const equityValues = {equity_values_json};
    const annotations = {annotations_json};
    const positions = {positions_json};
    const trades = {trade_history_json};

    // Render Equity Chart
    const ctx = document.getElementById('equityChart').getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(16, 185, 129, 0.35)');
    gradient.addColorStop(1, 'rgba(16, 185, 129, 0.0)');

    new Chart(ctx, {{
      type: 'line',
      data: {{
        labels: timePoints,
        datasets: [{{
          label: 'Total Equity ($)',
          data: equityValues,
          borderColor: '#10b981',
          borderWidth: 2.5,
          backgroundColor: gradient,
          fill: true,
          tension: 0.3,
          pointBackgroundColor: '#10b981',
          pointHoverRadius: 6,
          pointRadius: 4,
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            callbacks: {{
              label: function(context) {{
                return ' ' + annotations[context.dataIndex] + ' | Value: $' + context.parsed.y.toLocaleString();
              }}
            }}
          }}
        }},
        scales: {{
          x: {{
            grid: {{ color: 'rgba(51, 65, 85, 0.3)' }},
            ticks: {{ color: '#94a3b8', maxRotation: 45, minRotation: 0, font: {{ size: 10 }} }}
          }},
          y: {{
            grid: {{ color: 'rgba(51, 65, 85, 0.4)' }},
            ticks: {{
              color: '#94a3b8',
              callback: function(val) {{ return '$' + val.toLocaleString(); }}
            }}
          }}
        }}
      }}
    }});

    // Populate Positions
    const posTable = document.getElementById('positionsTable');
    const posLabels = [];
    const posValues = [];
    const posColors = ['#06b6d4', '#8b5cf6', '#f59e0b', '#ec4899', '#10b981'];

    positions.forEach((p, idx) => {{
      posLabels.push(p.ticker);
      posValues.push(p.market_value);

      const tr = document.createElement('tr');
      tr.className = 'hover:bg-slate-800/50 transition';
      tr.innerHTML = `
        <td class="py-3 font-semibold text-white flex items-center gap-2">
          <span class="w-2.5 h-2.5 rounded-full" style="background-color: ${{posColors[idx % posColors.length]}}"></span>
          ${{p.ticker}}
        </td>
        <td class="py-3 text-right text-slate-300 font-mono">${{p.shares.toFixed(2)}}</td>
        <td class="py-3 text-right text-slate-300 font-mono">$${{p.avg_price.toFixed(2)}}</td>
        <td class="py-3 text-right text-white font-mono font-semibold">$${{p.market_value.toLocaleString()}}</td>
        <td class="py-3 text-right text-slate-400 text-xs">${{p.entry_time || 'Active'}}</td>
      `;
      posTable.appendChild(tr);
    }});

    // Render Allocation Donut
    new Chart(document.getElementById('allocationChart'), {{
      type: 'doughnut',
      data: {{
        labels: posLabels,
        datasets: [{{
          data: posValues,
          backgroundColor: posColors.slice(0, posLabels.length),
          borderWidth: 0,
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ position: 'bottom', labels: {{ color: '#94a3b8', font: {{ size: 12 }} }} }}
        }}
      }}
    }});

    // Populate Trade History
    const tradesTable = document.getElementById('tradesTable');
    trades.slice().reverse().forEach(t => {{
      const tr = document.createElement('tr');
      tr.className = 'hover:bg-slate-800/50 transition';
      const badgeColor = t.action === 'BUY' ? 'bg-emerald-950 text-emerald-400 border-emerald-800' : 'bg-rose-950 text-rose-400 border-rose-800';
      tr.innerHTML = `
        <td class="py-3 text-slate-400 font-mono text-xs">${{t.timestamp.replace(' UTC', '')}}</td>
        <td class="py-3"><span class="px-2 py-0.5 rounded text-xs font-semibold border ${{badgeColor}}">${{t.action}}</span></td>
        <td class="py-3 font-bold text-white">${{t.ticker}}</td>
        <td class="py-3 text-right font-mono text-slate-300">${{t.shares.toFixed(2)}}</td>
        <td class="py-3 text-right font-mono text-slate-300">$${{t.exec_price.toFixed(2)}}</td>
        <td class="py-3 text-right font-mono font-semibold text-white">$${{t.total_value.toLocaleString()}}</td>
        <td class="py-3 text-slate-400 text-xs">${{t.reason || ''}}</td>
      `;
      tradesTable.appendChild(tr);
    }});
  </script>
</body>
</html>
"""
    return html_content


def main():
    parser = argparse.ArgumentParser(description="AI Trading Bot Intraday Dashboard")
    parser.add_argument("--portfolio", type=str, default="data/paper_portfolio.json", help="Path to portfolio file")
    parser.add_argument("--output", type=str, default="data/dashboard.html", help="Path to output HTML file")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    html = generate_dashboard_html(portfolio_file=args.portfolio)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    abs_path = os.path.abspath(args.output)
    print(f"\n[*] Generated Interactive Dashboard: {abs_path}")

    if not args.no_browser:
        print("[*] Opening in your web browser...")
        webbrowser.open(f"file:///{abs_path}")


if __name__ == "__main__":
    main()

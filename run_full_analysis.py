#!/usr/bin/env python3
"""
Download BTC/ETH hourly data (120h), plot, ask OpenAI for buy/sell/wait analysis,
and export separate plots and Markdown analysis files per asset.
"""

import csv
import re
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_DIR = Path("output")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from dotenv import load_dotenv
from openai import OpenAI
import requests

# Candlestick colors
COLOR_UP = "#26a69a"
COLOR_DOWN = "#ef5350"

API_URL = "https://api.binance.com/api/v3/klines"
INTERVAL = "1h"
LIMIT = 120


def download_symbol(symbol: str, output_dir: Path) -> Path:
    """Download hourly OHLCV data for the last 120 hours. Returns output path."""
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym = f"{sym}USDT"

    resp = requests.get(API_URL, params={"symbol": sym, "interval": INTERVAL, "limit": LIMIT})
    resp.raise_for_status()
    klines = resp.json()

    base = sym.replace("USDT", "").lower()
    out_path = output_dir / f"{base}_hourly_120h.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "datetime", "open", "high", "low", "close", "volume"])
        for k in klines:
            ts_ms, o, h, l, c, v = k[0], k[1], k[2], k[3], k[4], k[5]
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            writer.writerow([ts_ms, dt, o, h, l, c, v])

    print(f"Downloaded {len(klines)} hourly candles to {out_path}")
    return out_path


def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "datetime": datetime.strptime(row["datetime"].replace(" UTC", ""), "%Y-%m-%d %H:%M:%S"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })
    return rows


def plot_candlestick(ax, data, width_hours=0.6):
    dates = mdates.date2num([r["datetime"] for r in data])
    width = width_hours / 24

    for i, (d, row) in enumerate(zip(dates, data)):
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        is_up = c >= o
        color = COLOR_UP if is_up else COLOR_DOWN

        body_bottom = min(o, c)
        body_top = max(o, c)
        body_height = body_top - body_bottom
        if body_height == 0:
            body_height = (h - l) * 0.01

        rect = Rectangle(
            (d - width / 2, body_bottom),
            width,
            body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=1,
        )
        ax.add_patch(rect)
        ax.plot([d, d], [body_top, h], color=color, linewidth=1.5, solid_capstyle="round")
        ax.plot([d, d], [l, body_bottom], color=color, linewidth=1.5, solid_capstyle="round")

    ax.set_xlim(dates[0] - width, dates[-1] + width)
    ax.autoscale_view(scalex=False)


def build_summary_for_openai(btc: list[dict], eth: list[dict]) -> str:
    """Build a concise summary of the data for the OpenAI prompt."""
    def stats(data: list[dict], name: str) -> str:
        first, last = data[0], data[-1]
        low_120 = min(r["low"] for r in data)
        high_120 = max(r["high"] for r in data)
        return (
            f"{name}: Open={first['open']:.2f}, Close={last['close']:.2f}, "
            f"Low(120h)={low_120:.2f}, High(120h)={high_120:.2f}, "
            f"Range: {first['datetime']} - {last['datetime']}"
        )

    lines = [
        "Hourly OHLCV data for the last 120 hours:",
        stats(btc, "BTC"),
        stats(eth, "ETH"),
        "",
        "Recent hourly closes (last 24):",
        "BTC: " + ", ".join(f"{r['close']:.2f}" for r in btc[-24:]),
        "ETH: " + ", ".join(f"{r['close']:.2f}" for r in eth[-24:]),
    ]
    return "\n".join(lines)


def ask_openai(data_summary: str) -> str:
    load_dotenv()
    client = OpenAI()

    prompt = f"""You are a crypto trading analyst. Based on the following BTC and ETH hourly data (last 120 hours), assess whether there is any worthwhile probability for a successful BUY, SELL, or WAIT decision.

{data_summary}

Respond with two sections, one for each asset. Use this exact format:

## BTC
**BUY probability:** X% - [brief reason]
**SELL probability:** X% - [brief reason]
**WAIT probability:** X% - [brief reason]
**Stop loss:** [price in USD, e.g. 71000]
**Take profit:** [price in USD, e.g. 73500]
**Explanation:** [2-4 sentences for BTC]

## ETH
**BUY probability:** X% - [brief reason]
**SELL probability:** X% - [brief reason]
**WAIT probability:** X% - [brief reason]
**Stop loss:** [price in USD, e.g. 2050]
**Take profit:** [price in USD, e.g. 2250]
**Explanation:** [2-4 sentences for ETH]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )
    return response.choices[0].message.content


def parse_analysis_sections(analysis: str) -> tuple[str, str]:
    """Split analysis into BTC and ETH sections. Returns (btc_md, eth_md)."""
    parts = re.split(r"(?m)^##\s+", analysis.strip())
    btc_md = ""
    eth_md = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p.upper().startswith("BTC"):
            btc_md = "## " + p
        elif p.upper().startswith("ETH"):
            eth_md = "## " + p
    return btc_md, eth_md


def parse_sl_tp(text: str) -> tuple[float | None, float | None]:
    """Extract stop loss and take profit prices from analysis text. Returns (sl, tp)."""
    sl_match = re.search(r"\*\*Stop loss:\*\*\s*\$?([\d,.]+)", text, re.IGNORECASE)
    tp_match = re.search(r"\*\*Take profit:\*\*\s*\$?([\d,.]+)", text, re.IGNORECASE)
    sl = float(sl_match.group(1).replace(",", "")) if sl_match else None
    tp = float(tp_match.group(1).replace(",", "")) if tp_match else None
    return sl, tp


def create_single_asset_plot(
    data: list[dict], symbol: str, out_path: Path, sl: float | None = None, tp: float | None = None
):
    """Create candlestick + volume plot for one asset (no text overlay). Optionally draw SL/TP lines."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1]})
    date_fmt = mdates.DateFormatter("%d %b %H:%M")

    ax1, ax2 = axes[0], axes[1]
    plot_candlestick(ax1, data)
    if sl is not None:
        ax1.axhline(y=sl, color="#ef5350", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Stop loss: {sl:,.2f}")
    if tp is not None:
        ax1.axhline(y=tp, color="#26a69a", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Take profit: {tp:,.2f}")
    if sl is not None or tp is not None:
        ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title(f"{symbol} Price (H1)")
    ax1.set_ylabel("Price (USD)")
    ax1.xaxis.set_major_formatter(date_fmt)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    times = [r["datetime"] for r in data]
    colors = [COLOR_UP if r["close"] >= r["open"] else COLOR_DOWN for r in data]
    ax2.bar(times, [r["volume"] for r in data], color=colors, width=0.03, align="center")
    ax2.set_title(f"{symbol} Volume")
    ax2.set_ylabel("Volume")
    ax2.xaxis.set_major_formatter(date_fmt)
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    btc_dir = OUTPUT_DIR / "btc"
    eth_dir = OUTPUT_DIR / "eth"
    btc_dir.mkdir(parents=True, exist_ok=True)
    eth_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download data
    download_symbol("BTC", btc_dir)
    download_symbol("ETH", eth_dir)

    # 2. Load and build plot data
    btc = load_csv(btc_dir / "btc_hourly_120h.csv")
    eth = load_csv(eth_dir / "eth_hourly_120h.csv")

    # 3. Call OpenAI
    data_summary = build_summary_for_openai(btc, eth)
    print("\nCalling OpenAI API...")
    analysis = ask_openai(data_summary)
    print("\n--- OpenAI Analysis ---\n")
    print(analysis)

    # 4. Parse analysis and SL/TP
    btc_md, eth_md = parse_analysis_sections(analysis)
    btc_sl, btc_tp = parse_sl_tp(btc_md)
    eth_sl, eth_tp = parse_sl_tp(eth_md)

    # 5. Save separate plots with SL/TP lines
    create_single_asset_plot(btc, "BTC", btc_dir / "btc_plot.png", sl=btc_sl, tp=btc_tp)
    create_single_asset_plot(eth, "ETH", eth_dir / "eth_plot.png", sl=eth_sl, tp=eth_tp)
    print("Saved btc_plot.png, eth_plot.png")

    # 6. Export separate Markdown files
    (btc_dir / "btc_analysis.md").write_text(btc_md)
    (eth_dir / "eth_analysis.md").write_text(eth_md)
    print("\nSaved btc_analysis.md, eth_analysis.md (with stop loss and take profit)")
    print("Done.")


# TODO: extend the download time to the last 200 hours
# TODO: add the symbols open in MT5
# TODO: export a summary markdown table with decreasing probability of buy/sell
# TODO: run the script automaticlly every hour. 
if __name__ == "__main__":
    main()

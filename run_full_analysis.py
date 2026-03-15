#!/usr/bin/env python3
"""
Download BTC/ETH hourly data (200h), plot, ask OpenAI for buy/sell/wait analysis,
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
HOURS_LIMIT = 200
HOUR_MS = 3600 * 1000


def _kline_to_row(k: list) -> tuple[int, str, str, str, str, str, str]:
    ts_ms, o, h, l, c, v = int(k[0]), k[1], k[2], k[3], k[4], k[5]
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return (ts_ms, str(ts_ms), dt, o, h, l, c, v)


def _load_existing_csv(path: Path) -> list[tuple[int, str, str, str, str, str, str]] | None:
    """Load existing CSV and return rows as (ts_ms, ts_str, dt, o, h, l, c, v). Returns None if file missing."""
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            ts_ms = int(row["timestamp"])
            rows.append((ts_ms, row["timestamp"], row["datetime"], row["open"], row["high"], row["low"], row["close"], row["volume"]))
    return rows if rows else None


def download_symbol(symbol: str, output_dir: Path) -> Path:
    """Download hourly OHLCV data for the last 200 hours. Fetches only missing candles if file exists."""
    sym = symbol.upper()
    if not sym.endswith("USDT"):
        sym = f"{sym}USDT"

    base = sym.replace("USDT", "").lower()
    out_path = output_dir / f"{base}_hourly_200h.csv"

    # Support migration from old 120h filename
    legacy_path = output_dir / f"{base}_hourly_120h.csv"
    existing = _load_existing_csv(out_path) or _load_existing_csv(legacy_path)

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    current_hour_start = (now_ms // HOUR_MS) * HOUR_MS

    if existing is None:
        resp = requests.get(API_URL, params={"symbol": sym, "interval": INTERVAL, "limit": HOURS_LIMIT})
        resp.raise_for_status()
        klines = resp.json()
        rows = [_kline_to_row(k) for k in klines]
        total = len(rows)
    else:
        rows = list(existing)
        oldest_ts = rows[0][0]
        newest_ts = rows[-1][0]
        need_older = max(0, HOURS_LIMIT - len(rows))
        need_newer = max(0, (current_hour_start - newest_ts) // HOUR_MS)

        if need_older > 0:
            resp = requests.get(
                API_URL,
                params={"symbol": sym, "interval": INTERVAL, "endTime": oldest_ts - 1, "limit": need_older},
            )
            resp.raise_for_status()
            older = [_kline_to_row(k) for k in resp.json()]
            rows = sorted(older + rows, key=lambda r: r[0])

        if need_newer > 0:
            start_ts = newest_ts + HOUR_MS
            resp = requests.get(
                API_URL,
                params={"symbol": sym, "interval": INTERVAL, "startTime": start_ts, "limit": need_newer},
            )
            resp.raise_for_status()
            newer = [_kline_to_row(k) for k in resp.json()]
            rows = sorted(rows + newer, key=lambda r: r[0])

        # Deduplicate by timestamp
        seen = set()
        unique = []
        for r in rows:
            if r[0] not in seen:
                seen.add(r[0])
                unique.append(r)
        rows = unique[-HOURS_LIMIT:]  # Keep last 200
        total = len(rows)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "datetime", "open", "high", "low", "close", "volume"])
        for r in rows:
            writer.writerow([r[1], r[2], r[3], r[4], r[5], r[6], r[7]])

    print(f"Saved {total} hourly candles to {out_path}")
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
            f"Low(200h)={low_120:.2f}, High(200h)={high_120:.2f}, "
            f"Range: {first['datetime']} - {last['datetime']}"
        )

    lines = [
        "Hourly OHLCV data for the last 200 hours:",
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

    prompt = f"""You are a crypto trading analyst. Based on the following BTC and ETH hourly data (last 200 hours), assess whether there is any worthwhile probability for a successful BUY, SELL, or WAIT decision.

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


def parse_probabilities(text: str) -> tuple[int | None, int | None, int | None]:
    """Extract BUY, SELL, WAIT probabilities (0-100) from analysis text."""
    buy_m = re.search(r"\*\*BUY probability:\*\*\s*(\d+)%", text, re.IGNORECASE)
    sell_m = re.search(r"\*\*SELL probability:\*\*\s*(\d+)%", text, re.IGNORECASE)
    wait_m = re.search(r"\*\*WAIT probability:\*\*\s*(\d+)%", text, re.IGNORECASE)
    buy = int(buy_m.group(1)) if buy_m else None
    sell = int(sell_m.group(1)) if sell_m else None
    wait = int(wait_m.group(1)) if wait_m else None
    return buy, sell, wait


def build_summary_table(
    symbols_data: list[dict],
) -> str:
    """Build markdown table sorted by buy probability descending, then sell descending."""
    rows = sorted(
        symbols_data,
        key=lambda r: (-(r.get("buy") or 0), -(r.get("sell") or 0)),
    )
    lines = [
        "| Symbol | Buy % | Sell % | Wait % | Stop Loss | Take Profit |",
        "|--------|-------|--------|--------|-----------|-------------|",
    ]
    for r in rows:
        buy = f"{r['buy']}%" if r.get("buy") is not None else "-"
        sell = f"{r['sell']}%" if r.get("sell") is not None else "-"
        wait = f"{r['wait']}%" if r.get("wait") is not None else "-"
        sl = f"{r['sl']:,.2f}" if r.get("sl") is not None else "-"
        tp = f"{r['tp']:,.2f}" if r.get("tp") is not None else "-"
        lines.append(f"| {r['symbol']} | {buy} | {sell} | {wait} | {sl} | {tp} |")
    return "\n".join(lines)


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
    btc = load_csv(btc_dir / "btc_hourly_200h.csv")
    eth = load_csv(eth_dir / "eth_hourly_200h.csv")

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
    btc_buy, btc_sell, btc_wait = parse_probabilities(btc_md)
    eth_buy, eth_sell, eth_wait = parse_probabilities(eth_md)

    # 4b. Build and save summary table (sorted by buy descending, then sell descending)
    symbols_data = [
        {"symbol": "BTC", "buy": btc_buy, "sell": btc_sell, "wait": btc_wait, "sl": btc_sl, "tp": btc_tp},
        {"symbol": "ETH", "buy": eth_buy, "sell": eth_sell, "wait": eth_wait, "sl": eth_sl, "tp": eth_tp},
    ]
    summary_md = "# Trading Summary\n\n" + build_summary_table(symbols_data)
    (OUTPUT_DIR / "summary.md").write_text(summary_md)
    print("\nSaved summary.md")

    # 5. Save separate plots with SL/TP lines
    create_single_asset_plot(btc, "BTC", btc_dir / "btc_plot.png", sl=btc_sl, tp=btc_tp)
    create_single_asset_plot(eth, "ETH", eth_dir / "eth_plot.png", sl=eth_sl, tp=eth_tp)
    print("Saved btc_plot.png, eth_plot.png")

    # 6. Export separate Markdown files
    (btc_dir / "btc_analysis.md").write_text(btc_md)
    (eth_dir / "eth_analysis.md").write_text(eth_md)
    print("\nSaved btc_analysis.md, eth_analysis.md (with stop loss and take profit)")
    print("Done.")


# TODO: add the symbols open in MT5
# TODO: export a summary markdown table with decreasing probability of buy/sell
# TODO: run the script automaticlly every hour. 
if __name__ == "__main__":
    main()

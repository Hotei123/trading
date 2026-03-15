#!/usr/bin/env python3
"""
Download multi-asset hourly data (200h), plot, ask OpenAI for buy/sell/wait analysis,
and export separate plots and Markdown analysis files per asset.
"""

import csv
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

OUTPUT_DIR = Path("output")

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
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

# Symbol config: (display_name, source, yfinance_ticker or None for Binance)
# Forex/rates use 4 decimals; crypto/futures use 2
SYMBOL_CONFIG = [
    ("BTC", "binance", None),
    ("ETH", "binance", None),
    ("EURUSD", "yfinance", "EURUSD=X"),
    ("GBPUSD", "yfinance", "GBPUSD=X"),
    ("USDJPY", "yfinance", "JPY=X"),
    ("NQ", "yfinance", "NQ=F"),
    ("ES", "yfinance", "ES=F"),
    ("CL", "yfinance", "CL=F"),
    ("YM", "yfinance", "YM=F"),
    ("XAUUSD", "yfinance", "GC=F"),  # Gold futures (XAUUSD spot not on Yahoo)
    ("FDAX", "yfinance", "^GDAXI"),  # DAX index (FDAX futures proxy)
    ("NIY", "yfinance", "^N225"),   # Nikkei 225 index
    ("COFFEE", "yfinance", "KC=F"),
    ("SOYBEAN", "yfinance", "ZS=F"),
]
FOREX_RATE_SYMBOLS = frozenset({"EURUSD", "GBPUSD", "USDJPY"})  # use 4 decimals (rates)


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


def download_yfinance(symbol: str, ticker: str, output_dir: Path) -> Path:
    """Download hourly OHLCV for the last 200 hours via yfinance. Returns output path."""
    base = symbol.lower().replace("/", "").replace("^", "")
    out_path = output_dir / f"{base}_hourly_200h.csv"

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=HOURS_LIMIT + 24)  # extra buffer

    df = yf.download(ticker, start=start, end=end, interval="1h", progress=False, auto_adjust=True)
    if df.empty or len(df) < 2:
        raise RuntimeError(f"No data returned for {ticker} ({symbol})")

    # Flatten MultiIndex columns if present (e.g. from single-ticker download)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.tail(HOURS_LIMIT)
    rows = []
    for ts, row in df.iterrows():
        ts_utc = ts.tz_localize(timezone.utc) if ts.tzinfo is None else ts
        ts_ms = int(ts_utc.timestamp() * 1000)
        dt = ts_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        v = float(row.get("Volume", 0) or 0)
        rows.append((ts_ms, str(ts_ms), dt, str(o), str(h), str(l), str(c), str(v)))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "datetime", "open", "high", "low", "close", "volume"])
        for r in rows:
            writer.writerow([r[1], r[2], r[3], r[4], r[5], r[6], r[7]])

    print(f"Saved {len(rows)} hourly candles to {out_path}")
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


def build_summary_for_openai(assets: dict[str, list[dict]]) -> str:
    """Build a concise summary of the data for the OpenAI prompt."""
    def fmt_val(v: float, name: str) -> str:
        return f"{v:.4f}" if name in FOREX_RATE_SYMBOLS else f"{v:.2f}"

    def stats(data: list[dict], name: str) -> str:
        first, last = data[0], data[-1]
        low_200 = min(r["low"] for r in data)
        high_200 = max(r["high"] for r in data)
        return (
            f"{name}: Open={fmt_val(first['open'], name)}, Close={fmt_val(last['close'], name)}, "
            f"Low(200h)={fmt_val(low_200, name)}, High(200h)={fmt_val(high_200, name)}, "
            f"Range: {first['datetime']} - {last['datetime']}"
        )

    lines = ["Hourly OHLCV data for the last 200 hours:"]
    for name, data in assets.items():
        lines.append(stats(data, name))
    lines.append("")
    lines.append("Recent hourly closes (last 24):")
    for name, data in assets.items():
        lines.append(f"{name}: " + ", ".join(fmt_val(r["close"], name) for r in data[-24:]))
    return "\n".join(lines)


def ask_openai(data_summary: str, symbols: list[str]) -> str:
    load_dotenv()
    client = OpenAI()

    sections = []
    for sym in symbols:
        if sym in FOREX_RATE_SYMBOLS:
            sl_tp_hint = "**Stop loss:** [rate, e.g. 1.0650]\n**Take profit:** [rate, e.g. 1.0950]"
        else:
            sl_tp_hint = "**Stop loss:** [price in USD]\n**Take profit:** [price in USD]"
        sections.append(
            f"## {sym}\n"
            f"**BUY probability:** X% - [brief reason]\n"
            f"**SELL probability:** X% - [brief reason]\n"
            f"**WAIT probability:** X% - [brief reason]\n"
            f"{sl_tp_hint}\n"
            f"**Explanation:** [2-4 sentences for {sym}]"
        )

    prompt = f"""You are a trading analyst. Based on the following hourly data (last 200 hours), assess whether there is any worthwhile probability for a successful BUY, SELL, or WAIT decision for each asset.
- For forex (EURUSD, GBPUSD, USDJPY): BUY = long base currency, SELL = short base currency.
- For XAUUSD (gold): BUY = long gold, SELL = short gold.
- For futures (NQ, ES, CL, YM, FDAX, NIY, COFFEE, SOYBEAN): BUY = long, SELL = short.

{data_summary}

Respond with one section per asset. Use this exact format:

"""
    prompt += "\n\n".join(sections)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
    )
    return response.choices[0].message.content


def parse_analysis_sections(analysis: str, symbols: list[str]) -> dict[str, str]:
    """Split analysis into sections per symbol. Returns {symbol: md}."""
    parts = re.split(r"(?m)^##\s+", analysis.strip())
    result = {}
    syms_sorted = sorted(symbols, key=len, reverse=True)  # match longer first (e.g. USDJPY before USD)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        for sym in syms_sorted:
            if p.upper().startswith(sym.upper()):
                result[sym] = "## " + p
                break
    return result


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
    """Build markdown table sorted by max(buy, sell) descending, then by config order for ties."""
    config_order = {c[0]: i for i, c in enumerate(SYMBOL_CONFIG)}
    rows = sorted(
        symbols_data,
        key=lambda r: (-(max(r.get("buy") or 0, r.get("sell") or 0)), config_order.get(r["symbol"], 999)),
    )
    lines = [
        "| Symbol | Buy % | Sell % | Wait % | Stop Loss | Take Profit |",
        "|--------|-------|--------|--------|-----------|-------------|",
    ]
    for r in rows:
        buy = f"{r['buy']}%" if r.get("buy") is not None else "-"
        sell = f"{r['sell']}%" if r.get("sell") is not None else "-"
        wait = f"{r['wait']}%" if r.get("wait") is not None else "-"
        use_4dec = r.get("symbol") in FOREX_RATE_SYMBOLS
        sl = f"{r['sl']:,.4f}" if r.get("sl") is not None and use_4dec else (f"{r['sl']:,.2f}" if r.get("sl") is not None else "-")
        tp = f"{r['tp']:,.4f}" if r.get("tp") is not None and use_4dec else (f"{r['tp']:,.2f}" if r.get("tp") is not None else "-")
        lines.append(f"| {r['symbol']} | {buy} | {sell} | {wait} | {sl} | {tp} |")
    return "\n".join(lines)


def create_single_asset_plot(
    data: list[dict], symbol: str, out_path: Path, sl: float | None = None, tp: float | None = None
):
    """Create candlestick + volume plot for one asset (no text overlay). Optionally draw SL/TP lines."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1]})
    date_fmt = mdates.DateFormatter("%d %b %H:%M")
    is_forex = symbol in FOREX_RATE_SYMBOLS

    ax1, ax2 = axes[0], axes[1]
    plot_candlestick(ax1, data)
    if sl is not None:
        ax1.axhline(y=sl, color="#ef5350", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Stop loss: {sl:,.4f}" if is_forex else f"Stop loss: {sl:,.2f}")
    if tp is not None:
        ax1.axhline(y=tp, color="#26a69a", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Take profit: {tp:,.4f}" if is_forex else f"Take profit: {tp:,.2f}")
    if sl is not None or tp is not None:
        ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title(f"{symbol} Price (H1)")
    ax1.set_ylabel("Rate" if is_forex else "Price (USD)")
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
    symbols = [c[0] for c in SYMBOL_CONFIG]
    dirs = {s: OUTPUT_DIR / s.lower().replace("/", "").replace("^", "") for s in symbols}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # 1. Download data
    for sym, source, ticker in SYMBOL_CONFIG:
        out_dir = dirs[sym]
        if source == "binance":
            download_symbol(sym, out_dir)
        else:
            try:
                download_yfinance(sym, ticker, out_dir)
            except Exception as e:
                print(f"Warning: {sym} ({ticker}): {e}")

    # 2. Load and build plot data (only successfully downloaded)
    assets = {}
    for sym, source, ticker in SYMBOL_CONFIG:
        base = sym.lower().replace("/", "").replace("^", "")
        csv_path = dirs[sym] / f"{base}_hourly_200h.csv"
        if csv_path.exists():
            assets[sym] = load_csv(csv_path)

    # 3. Call OpenAI
    data_summary = build_summary_for_openai(assets)
    print("\nCalling OpenAI API...")
    analysis = ask_openai(data_summary, list(assets.keys()))
    print("\n--- OpenAI Analysis ---\n")
    print(analysis)

    # 4. Parse analysis and SL/TP
    sections = parse_analysis_sections(analysis, list(assets.keys()))
    symbols_data = []
    for sym in assets:
        md = sections.get(sym, "")
        sl, tp = parse_sl_tp(md)
        buy, sell, wait = parse_probabilities(md)
        symbols_data.append({"symbol": sym, "buy": buy, "sell": sell, "wait": wait, "sl": sl, "tp": tp})

    # 4b. Build and save summary table
    summary_md = "# Trading Summary\n\n" + build_summary_table(symbols_data)
    (OUTPUT_DIR / "summary.md").write_text(summary_md)
    print("\nSaved summary.md")

    # 5. Save separate plots with SL/TP lines
    for sym in assets:
        data = assets[sym]
        sd = next(r for r in symbols_data if r["symbol"] == sym)
        create_single_asset_plot(data, sym, dirs[sym] / f"{sym.lower()}_plot.png", sl=sd["sl"], tp=sd["tp"])
    print("Saved plots")

    # 6. Export separate Markdown files
    for sym in assets:
        (dirs[sym] / f"{sym.lower()}_analysis.md").write_text(sections.get(sym, ""))
    print("Saved analysis markdown files")
    print("Done.")


# TODO: add the symbols open in MT5
# TODO: run the script automaticlly every hour. 
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot BTC and ETH hourly OHLCV data."""

import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def load_csv(path):
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

def main():
    btc = load_csv("btc_hourly_12h.csv")
    eth = load_csv("eth_hourly_12h.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})

    # BTC price
    ax1 = axes[0, 0]
    times = [r["datetime"] for r in btc]
    ax1.fill_between(times, [r["low"] for r in btc], [r["high"] for r in btc], alpha=0.2, color="steelblue")
    ax1.plot(times, [r["close"] for r in btc], "o-", color="steelblue", linewidth=2, markersize=6)
    ax1.set_title("BTC Price (H1)")
    ax1.set_ylabel("Price (USD)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    # ETH price
    ax2 = axes[0, 1]
    times_eth = [r["datetime"] for r in eth]
    ax2.fill_between(times_eth, [r["low"] for r in eth], [r["high"] for r in eth], alpha=0.2, color="forestgreen")
    ax2.plot(times_eth, [r["close"] for r in eth], "o-", color="forestgreen", linewidth=2, markersize=6)
    ax2.set_title("ETH Price (H1)")
    ax2.set_ylabel("Price (USD)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)

    # BTC volume
    ax3 = axes[1, 0]
    colors = ["#2ecc71" if r["close"] >= r["open"] else "#e74c3c" for r in btc]
    ax3.bar(times, [r["volume"] for r in btc], color=colors, width=0.03)
    ax3.set_title("BTC Volume")
    ax3.set_ylabel("Volume")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3, axis="y")

    # ETH volume
    ax4 = axes[1, 1]
    colors_eth = ["#2ecc71" if r["close"] >= r["open"] else "#e74c3c" for r in eth]
    ax4.bar(times_eth, [r["volume"] for r in eth], color=colors_eth, width=0.03)
    ax4.set_title("ETH Volume")
    ax4.set_ylabel("Volume")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("btc_eth_plot.png", dpi=150, bbox_inches="tight")
    print("Saved btc_eth_plot.png")
    plt.close()

if __name__ == "__main__":
    main()

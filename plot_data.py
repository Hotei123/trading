#!/usr/bin/env python3
"""Plot BTC and ETH hourly OHLCV data with Japanese candlesticks and volume."""

import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime

# Candlestick colors
COLOR_UP = "#26a69a"   # green (bullish)
COLOR_DOWN = "#ef5350"  # red (bearish)


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


def plot_candlestick(ax, data, width_hours=0.6):
    """Draw Japanese candlesticks on the given axes."""
    dates = mdates.date2num([r["datetime"] for r in data])
    width = width_hours / 24  # convert hours to day fraction

    for i, (d, row) in enumerate(zip(dates, data)):
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        is_up = c >= o
        color = COLOR_UP if is_up else COLOR_DOWN

        body_bottom = min(o, c)
        body_top = max(o, c)
        body_height = body_top - body_bottom
        if body_height == 0:
            body_height = (h - l) * 0.01  # doji: thin body

        # Body
        rect = Rectangle(
            (d - width / 2, body_bottom),
            width,
            body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=1,
        )
        ax.add_patch(rect)

        # Upper wick
        ax.plot([d, d], [body_top, h], color=color, linewidth=1.5, solid_capstyle="round")
        # Lower wick
        ax.plot([d, d], [l, body_bottom], color=color, linewidth=1.5, solid_capstyle="round")

    ax.set_xlim(dates[0] - width, dates[-1] + width)
    ax.autoscale_view(scalex=False)


def main():
    btc = load_csv("btc_hourly_12h.csv")
    eth = load_csv("eth_hourly_12h.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]})
    date_fmt = mdates.DateFormatter("%d %b %H:%M")

    # BTC candlesticks
    ax1 = axes[0, 0]
    plot_candlestick(ax1, btc)
    ax1.set_title("BTC Price (H1)")
    ax1.set_ylabel("Price (USD)")
    ax1.xaxis.set_major_formatter(date_fmt)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    # ETH candlesticks
    ax2 = axes[0, 1]
    plot_candlestick(ax2, eth)
    ax2.set_title("ETH Price (H1)")
    ax2.set_ylabel("Price (USD)")
    ax2.xaxis.set_major_formatter(date_fmt)
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)

    # BTC volume
    ax3 = axes[1, 0]
    times = [r["datetime"] for r in btc]
    colors = [COLOR_UP if r["close"] >= r["open"] else COLOR_DOWN for r in btc]
    ax3.bar(times, [r["volume"] for r in btc], color=colors, width=0.03, align="center")
    ax3.set_title("BTC Volume")
    ax3.set_ylabel("Volume")
    ax3.xaxis.set_major_formatter(date_fmt)
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3, axis="y")

    # ETH volume
    ax4 = axes[1, 1]
    times_eth = [r["datetime"] for r in eth]
    colors_eth = [COLOR_UP if r["close"] >= r["open"] else COLOR_DOWN for r in eth]
    ax4.bar(times_eth, [r["volume"] for r in eth], color=colors_eth, width=0.03, align="center")
    ax4.set_title("ETH Volume")
    ax4.set_ylabel("Volume")
    ax4.xaxis.set_major_formatter(date_fmt)
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("btc_eth_plot.png", dpi=150, bbox_inches="tight")
    print("Saved btc_eth_plot.png")
    plt.close()


if __name__ == "__main__":
    main()

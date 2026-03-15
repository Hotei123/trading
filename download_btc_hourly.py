#!/usr/bin/env python3
"""Download BTC hourly OHLCV data for the last 12 hours from Binance."""

import csv
import requests
from datetime import datetime

API_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
LIMIT = 12

def main():
    resp = requests.get(API_URL, params={
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": LIMIT,
    })
    resp.raise_for_status()
    klines = resp.json()

    out_path = "btc_hourly_12h.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "datetime", "open", "high", "low", "close", "volume"])
        for k in klines:
            ts_ms, o, h, l, c, v = k[0], k[1], k[2], k[3], k[4], k[5]
            dt = datetime.utcfromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M:%S UTC")
            writer.writerow([ts_ms, dt, o, h, l, c, v])

    print(f"Downloaded {len(klines)} hourly candles to {out_path}")
    print(f"Range: {datetime.utcfromtimestamp(klines[0][0]/1000)} - {datetime.utcfromtimestamp(klines[-1][0]/1000)} UTC")

if __name__ == "__main__":
    main()

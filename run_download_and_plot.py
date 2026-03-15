#!/usr/bin/env python3
"""Run data download and plot in sequence."""

import subprocess
import sys

def main():
    subprocess.run([sys.executable, "download_crypto_hourly.py", "BTC"], check=True)
    subprocess.run([sys.executable, "download_crypto_hourly.py", "ETH"], check=True)
    subprocess.run([sys.executable, "plot_data.py"], check=True)
    print("Done: data downloaded and plot saved.")

if __name__ == "__main__":
    main()

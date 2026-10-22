import os
import pandas as pd
import yfinance as yf

# --------------------------------------------------
# Read the acquisitions dataset
# --------------------------------------------------
deals = pd.read_csv("data/raw/acquisitions.csv")

# Create folder if it doesn't exist
os.makedirs("data/stock_data", exist_ok=True)

# Get unique ticker symbols
tickers = deals["Ticker"].unique()

print(f"Found {len(tickers)} unique companies.\n")

# --------------------------------------------------
# Download stock data for each company
# --------------------------------------------------
for ticker in tickers:

    print(f"Downloading {ticker}...")

    try:
        stock = yf.download(
            ticker,
            start="2000-01-01",
            end="2025-01-01",
            auto_adjust=False,
            progress=False
        )

        # Remove MultiIndex created by yfinance
        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.droplevel(1)

        # Move Date index into a normal column
        stock.reset_index(inplace=True)

        # Save one CSV per company
        if stock.empty:
         print(f"Skipping {ticker} (no data)")
         continue
        stock.to_csv(f"data/stock_data/{ticker}.csv", index=False)

        print(f"✓ Saved {ticker}.csv")

    except Exception as e:
        print(f"✗ Failed for {ticker}")
        print(e)

print("\nAll downloads completed!")
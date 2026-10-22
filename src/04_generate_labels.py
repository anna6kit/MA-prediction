import os
import pandas as pd

# -----------------------------
# Load acquisitions
# -----------------------------
deals = pd.read_csv("data/raw/acquisitions.csv")

# Store generated labels
returns = []
labels = []

# ---------------------------------------------------
# Process every acquisition
# ---------------------------------------------------
for _, row in deals.iterrows():

    ticker = row["Ticker"]
    deal_date = pd.to_datetime(row["Date"])

    stock_file = f"data/stock_data/{ticker}.csv"

    # Skip if stock file doesn't exist
    if not os.path.exists(stock_file):
        returns.append(None)
        labels.append(None)
        continue

    # Read stock prices
    stock = pd.read_csv(stock_file)
    if "Date" not in stock.columns:
     print(f"Skipping {ticker} - invalid stock file")
     returns.append(None)
     labels.append(None)
     continue


    # Convert Date column into datetime
    stock["Date"] = pd.to_datetime(stock["Date"])

    # Keep only rows on or after acquisition date
    stock = stock[stock["Date"] >= deal_date]

    # Skip if no stock data available
    if len(stock) < 31:
        returns.append(None)
        labels.append(None)
        continue

    # Closing price on acquisition date (or next trading day)
    current_close = stock.iloc[0]["Close"]

    # Closing price roughly 30 trading days later
    future_close = stock.iloc[30]["Close"]

    # Calculate percentage return
    pct_return = ((future_close - current_close) / current_close) * 100

    returns.append(round(pct_return, 2))

    # Generate binary label
    if pct_return > 0:
        labels.append(1)
    else:
        labels.append(0)

# ---------------------------------------
# Add new columns
# ---------------------------------------
deals["Return (%)"] = returns
deals["Label"] = labels

# Create folder if needed
os.makedirs("data/final", exist_ok=True)

# Save final dataset
deals.to_csv("data/final/labeled_dataset.csv", index=False)

print(deals)
print("\nDataset saved successfully!")
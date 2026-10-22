import pandas as pd
import yfinance as yf

# -------------------------------------
# Load acquisition dataset
# -------------------------------------
deals = pd.read_csv("data/final/labeled_dataset.csv")

# Store fetched information
market_caps = []
sectors = []
industries = []

# Avoid downloading the same company twice
cache = {}

# -------------------------------------
# Fetch company information
# -------------------------------------
for ticker in deals["Ticker"]:

    # Use cached data if already fetched
    if ticker in cache:
        info = cache[ticker]
    else:
        print(f"Fetching {ticker}...")

        try:
            company = yf.Ticker(ticker)
            info = company.info
            cache[ticker] = info

        except Exception:
            info = {}

    # Market Capitalization
    market_caps.append(info.get("marketCap"))

    # Business Sector
    sectors.append(info.get("sector"))

    # Industry
    industries.append(info.get("industry"))

# -------------------------------------
# Add new columns
# -------------------------------------
deals["Market Cap"] = market_caps
deals["Sector"] = sectors
deals["Industry"] = industries

# Save updated dataset
deals.to_csv("data/final/ml_dataset.csv", index=False)

print("\nDataset updated successfully!")
print(deals.head())
import pandas as pd

# Load acquisition dataset
deals = pd.read_csv("data/sample_deals.csv")

# Load stock prices
stock = pd.read_csv("data/google_stock.csv")

print(stock.columns)
# Convert date columns into datetime format
stock["Date"] = pd.to_datetime(stock["Date"])


# Add day "1" to dates like "April 2003"
deals["date"] = deals["date"].apply(
    lambda x: "1 " + x if pd.notna(x) and len(str(x).split()) == 2 else x
)

deals["date"] = pd.to_datetime(deals["date"], errors="coerce")

# Keep only the columns we need
stock = stock[["Date", "Close"]]

labels = []

# -------------------------------------------------
# Generate one label for every acquisition
# -------------------------------------------------
for _, row in deals.iterrows():

    deal_date = row["date"]

    # Find first trading day after acquisition
    after = stock[stock["Date"] >= deal_date]

    if len(after) == 0:
        labels.append(None)
        continue

    current_price = after.iloc[0]["Close"]

    # Trading day roughly 30 calendar days later
    future = after.iloc[min(30, len(after)-1)]["Close"]

    # Label:
    # 1 -> Stock increased
    # 0 -> Stock decreased
    if future > current_price:
        labels.append(1)
    else:
        labels.append(0)

# Add labels to dataset
deals["label"] = labels

# Save new dataset
deals.to_csv("data/labeled_deals.csv", index=False)

print(deals.head())
print("\nSaved -> data/labeled_deals.csv")
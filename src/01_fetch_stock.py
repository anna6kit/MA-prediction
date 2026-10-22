import yfinance as yf

ticker = "GOOG"

stock = yf.download(
    ticker,
    start="2000-01-01",
    end="2025-01-01",
    auto_adjust=False
)

# Remove the ticker level from the columns
stock.columns = stock.columns.droplevel(1)

# Convert Date index into a normal column
stock = stock.reset_index()

print(stock.head())

stock.to_csv("data/google_stock.csv", index=False)

print("Done!")
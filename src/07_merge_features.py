import pandas as pd

# ----------------------------------------------------
# Load the two datasets
# ----------------------------------------------------

# Main ML dataset (contains stock-based features and labels)
ml_data = pd.read_csv("data/final/ml_dataset.csv")

# Deal-specific features collected separately
deal_features = pd.read_csv("data/raw/deal_features.csv")

# ----------------------------------------------------
# Merge datasets
# ----------------------------------------------------
# We merge using Acquirer and Target because together
# they uniquely identify each acquisition.

merged = pd.merge(
    ml_data,
    deal_features,
    on=["Acquirer", "Target"],
    how="left"      # Keep all acquisitions even if some deal info is missing
)

# ----------------------------------------------------
# Save merged dataset
# ----------------------------------------------------

merged.to_csv("data/final/final_dataset.csv", index=False)

print("Merge Successful!\n")

print("Shape :", merged.shape)

print("\nColumns:\n")
print(merged.columns)

print("\nFirst 5 Rows:\n")
print(merged.head())
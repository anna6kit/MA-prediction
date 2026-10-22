import pandas as pd

# Load the dataset
df = pd.read_csv("data/mergers and acquisitions.csv")

print("===== First 5 Rows =====")
print(df.head())

print("\n===== Dataset Shape =====")
print(df.shape)

print("\n===== Column Names =====")
print(df.columns)

print("\n===== Dataset Information =====")
print(df.info())

print("\n===== Missing Values =====")
print(df.isnull().sum())

print("\n===== Statistical Summary =====")
print(df.describe())
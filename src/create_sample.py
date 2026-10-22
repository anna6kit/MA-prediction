import pandas as pd

df = pd.read_csv("data/mergers and acquisitions.csv")

sample = df.head(20)

sample.to_csv("data/sample_deals.csv", index=False)

print(sample)
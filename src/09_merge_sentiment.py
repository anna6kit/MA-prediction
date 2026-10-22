import pandas as pd

financial = pd.read_csv("data/final/final_dataset.csv")
sentiment = pd.read_csv("data/final/sentiment_scores.csv")

ml = financial.merge(
    sentiment[
        [
            "Acquirer",
            "Target",
            "PredictedSentiment",
            "PositiveScore",
            "NegativeScore",
            "NeutralScore",
        ]
    ],
    on=["Acquirer", "Target"],
    how="left",
)

ml.to_csv("data/final/ml_dataset.csv", index=False)

print(ml.head())
print("\nMerged successfully!")
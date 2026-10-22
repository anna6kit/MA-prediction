import pandas as pd
from transformers import pipeline

# ---------------------------------------------------
# Model Name
# ---------------------------------------------------
MODEL_NAME = "ProsusAI/finbert"

# ---------------------------------------------------
# Load FinBERT Pipeline
# ---------------------------------------------------
print("Loading FinBERT model...")

classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    top_k=None  # Return scores for all sentiment classes
)

print("Model Loaded!\n")

# ---------------------------------------------------
# Load Headlines
# ---------------------------------------------------
headlines = pd.read_csv("data/raw/headlines.csv")

# ---------------------------------------------------
# Lists to store results
# ---------------------------------------------------
positive_scores = []
negative_scores = []
neutral_scores = []
predicted_sentiment = []

# ---------------------------------------------------
# Run FinBERT
# ---------------------------------------------------
for headline in headlines["Headline"]:

    results = classifier(headline)[0]

    # Convert list of dictionaries into a dictionary
    scores = {item["label"].lower(): item["score"] for item in results}

    # Get predicted sentiment (highest probability)
    predicted = max(results, key=lambda x: x["score"])["label"].lower()

    predicted_sentiment.append(predicted)
    positive_scores.append(scores.get("positive", 0.0))
    negative_scores.append(scores.get("negative", 0.0))
    neutral_scores.append(scores.get("neutral", 0.0))

# ---------------------------------------------------
# Save Results
# ---------------------------------------------------
headlines["PredictedSentiment"] = predicted_sentiment
headlines["PositiveScore"] = positive_scores
headlines["NegativeScore"] = negative_scores
headlines["NeutralScore"] = neutral_scores

headlines.to_csv(
    "data/final/sentiment_scores.csv",
    index=False
)

# ---------------------------------------------------
# Display Sample Output
# ---------------------------------------------------
print(headlines.head())

print("\nSentiment scores saved successfully!")
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
data = pd.read_csv("data/final/final_dataset.csv")

# ---------------------------------------------------
# Convert Deal Value to Numeric
# ---------------------------------------------------
data["DealValueUSD_Million"] = pd.to_numeric(
    data["DealValueUSD_Million"],
    errors="coerce"
)

# ---------------------------------------------------
# Remove Missing Values
# ---------------------------------------------------
data = data.dropna()

# ---------------------------------------------------
# Encode Categorical Features
# ---------------------------------------------------
encoder = LabelEncoder()

categorical_columns = [
    "Acquirer",
    "Sector",
    "Industry",
    "TargetIndustry",
    "TargetCountry"
]

for col in categorical_columns:
    data[col] = encoder.fit_transform(data[col])

# ---------------------------------------------------
# Features
# ---------------------------------------------------
X = data[
    [
        "Acquirer",
        "Market Cap",
        "Sector",
        "Industry",
        "DealValueUSD_Million",
        "TargetIndustry",
        "TargetCountry"
    ]
]

# ---------------------------------------------------
# Target
# ---------------------------------------------------
y = data["Label"]

# ---------------------------------------------------
# Random Forest Model
# ---------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ---------------------------------------------------
# 5-Fold Cross Validation
# ---------------------------------------------------


accuracy = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="accuracy"
)

precision = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring=make_scorer(precision_score, zero_division=0)
)

recall = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring=make_scorer(recall_score, zero_division=0)
)

f1 = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring=make_scorer(f1_score, zero_division=0)
)

# ---------------------------------------------------
# Train on Entire Dataset
# ---------------------------------------------------
model.fit(X, y)

# ---------------------------------------------------
# Feature Importance
# ---------------------------------------------------
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

# ---------------------------------------------------
# Print Results
# ---------------------------------------------------
print("\n===================================")
print("5-Fold Cross Validation Results")
print("===================================")

print(f"\nAccuracy : {accuracy.mean():.3f}")
print(f"Precision: {precision.mean():.3f}")
print(f"Recall   : {recall.mean():.3f}")
print(f"F1 Score : {f1.mean():.3f}")

print("\nFold-wise Accuracy:")
print(accuracy)

print("\n===================================")
print("Feature Importance")
print("===================================\n")

print(importance)
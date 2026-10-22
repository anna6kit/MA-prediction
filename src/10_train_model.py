import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ===================================================
# Load Dataset
# ===================================================

print("Loading dataset...\n")

df = pd.read_csv("data/final/ml_dataset.csv")

print("Dataset Shape:", df.shape)

# ===================================================
# Features & Label
# ===================================================

drop_columns = [
    "Date",
    "Ticker",
    "Return (%)",
    "PredictedSentiment",
]

X = df.drop(columns=drop_columns + ["Label"])
y = df["Label"]

# ===================================================
# Feature Types
# ===================================================

numeric_features = [
    "Market Cap",
    "DealValueUSD_Million",
    "PositiveScore",
    "NegativeScore",
    "NeutralScore",
]

categorical_features = [
    "Acquirer",
    "Sector",
    "Industry",
    "TargetIndustry",
    "TargetCountry",
]

# ===================================================
# Preprocessing
# ===================================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ===================================================
# Random Forest Model
# ===================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model),
])

# ===================================================
# TRAIN - TEST SPLIT
# ===================================================

print("\n========================================")
print("TRAIN - TEST SPLIT RESULTS")
print("========================================\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.title("Confusion Matrix (Train-Test Split)")

plt.show()

# ===================================================
# STRATIFIED 5-FOLD CROSS VALIDATION
# ===================================================

print("\n========================================")
print("STRATIFIED 5-FOLD CROSS VALIDATION")
print("========================================\n")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}

scores = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring,
)

for metric in scoring.keys():

    values = scores[f"test_{metric}"]

    print(
        f"{metric.capitalize():10}: "
        f"{values.mean():.3f} ± {values.std():.3f}"
    )

# ===================================================
# FEATURE IMPORTANCE
# ===================================================

print("\n========================================")
print("FEATURE IMPORTANCE")
print("========================================\n")

pipeline.fit(X, y)

feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = pipeline.named_steps[
    "classifier"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance,
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False,
)

print(importance_df.head(15))

# ===================================================
# FEATURE IMPORTANCE GRAPH
# ===================================================

top_features = importance_df.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 10 Important Features")

plt.tight_layout()

plt.show()

# ===================================================
# SAVE MODEL
# ===================================================

joblib.dump(
    pipeline,
    "models/random_forest_model.pkl"
)

print("\nModel saved to models/random_forest_model.pkl")

print("\n========================================")
print("TRAINING COMPLETED SUCCESSFULLY!")
print("========================================")
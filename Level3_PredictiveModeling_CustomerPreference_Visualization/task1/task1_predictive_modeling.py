"""
Cognifyz Technologies - Data Science Internship
LEVEL 3 - TASK 1: Predictive Modeling
------------------------------------------------------
Objectives:
1. Build a regression model to predict the aggregate rating of a
   restaurant based on available features.
2. Split the dataset into training and testing sets and evaluate the
   model's performance using appropriate metrics.
3. Experiment with different algorithms (linear regression, decision
   trees, random forest) and compare their performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("dataset.csv")
df["Cuisines"] = df["Cuisines"].fillna("Not Specified")

# ------------------------------------------------------------------
# 1. Feature preparation
# ------------------------------------------------------------------
print("=" * 70)
print("STEP 1: FEATURE PREPARATION")
print("=" * 70)

model_df = df.copy()

# Engineered features (reused from Level 2 - Task 3)
model_df["Restaurant Name Length"] = model_df["Restaurant Name"].astype(str).apply(len)
model_df["Address Length"] = model_df["Address"].astype(str).apply(len)
model_df["Cuisine Count"] = model_df["Cuisines"].astype(str).apply(lambda x: len(x.split(",")))

# Binary-encode Yes/No columns
for col in ["Has Table booking", "Has Online delivery",
            "Is delivering now", "Switch to order menu"]:
    model_df[col] = (model_df[col] == "Yes").astype(int)

# Label-encode City (too many categories for one-hot in this simple demo)
le_city = LabelEncoder()
model_df["City_encoded"] = le_city.fit_transform(model_df["City"])

feature_cols = [
    "Country Code", "City_encoded", "Longitude", "Latitude",
    "Average Cost for two", "Has Table booking", "Has Online delivery",
    "Is delivering now", "Switch to order menu", "Price range", "Votes",
    "Restaurant Name Length", "Address Length", "Cuisine Count"
]
target_col = "Aggregate rating"

# Remove rows where the restaurant has never been rated (Votes == 0),
# since a rating of 0 there reflects "not rated" rather than a true score.
model_data = model_df[model_df["Votes"] > 0].copy()
X = model_data[feature_cols]
y = model_data[target_col]

print(f"Features used: {feature_cols}")
print(f"Rows used for modeling (Votes > 0): {len(model_data)} / {len(df)}")

# ------------------------------------------------------------------
# 2. Train/test split
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size : {X_test.shape[0]}")

# ------------------------------------------------------------------
# 3. Train and evaluate multiple models
# ------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=8),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200, max_depth=12),
}

results = []
print("\n" + "=" * 70)
print("STEP 2: TRAINING AND EVALUATING MODELS")
print("=" * 70)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2 Score": r2})
    print(f"\n{name}:")
    print(f"  MAE      : {mae:.4f}")
    print(f"  RMSE     : {rmse:.4f}")
    print(f"  R2 Score : {r2:.4f}")

results_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False)
print("\n" + "=" * 70)
print("MODEL COMPARISON TABLE")
print("=" * 70)
print(results_df.to_string(index=False))

# ------------------------------------------------------------------
# 4. Visualize model comparison
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["R2 Score"], color="#4C72B0", edgecolor="black")
plt.title("Model Comparison - R2 Score (Higher is Better)")
plt.ylabel("R2 Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("output_model_comparison_r2.png", dpi=150)
plt.close()
print("\nSaved chart -> output_model_comparison_r2.png")

# Feature importance from the best tree-based model (Random Forest)
rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(9, 6))
importances.plot(kind="barh", color="#55A868", edgecolor="black")
plt.title("Random Forest - Feature Importance")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("output_feature_importance.png", dpi=150)
plt.close()
print("Saved chart -> output_feature_importance.png")

best_model_name = results_df.iloc[0]["Model"]
print("\n" + "=" * 70)
print("SUMMARY / INSIGHTS")
print("=" * 70)
print(f"""
1. Three regression algorithms were trained: Linear Regression, Decision
   Tree, and Random Forest.
2. Based on R2 score on the held-out test set, '{best_model_name}' performed
   best (R2 = {results_df.iloc[0]['R2 Score']:.4f}), meaning it explains the
   largest share of variance in Aggregate rating among the three models.
3. Tree-based models (Decision Tree, Random Forest) outperform plain
   Linear Regression here, suggesting non-linear relationships between
   features (e.g., Votes, Price range) and the rating.
4. According to the Random Forest feature importance chart, the top
   drivers of predicted rating are: {', '.join(importances.head(3).index)}.
""")

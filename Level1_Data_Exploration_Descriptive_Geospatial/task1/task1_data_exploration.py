"""
Cognifyz Technologies - Data Science Internship
LEVEL 1 - TASK 1: Data Exploration and Preprocessing
------------------------------------------------------
Objectives:
1. Explore the dataset and identify the number of rows and columns.
2. Check for missing values in each column and handle them accordingly.
3. Perform data type conversion if necessary.
4. Analyze the distribution of the target variable ("Aggregate rating")
   and identify any class imbalances.
"""

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 0. Load the dataset
# ------------------------------------------------------------------
df = pd.read_csv("dataset.csv")

print("=" * 70)
print("STEP 1: BASIC SHAPE OF THE DATASET")
print("=" * 70)
print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")
print("\nColumn names:")
print(list(df.columns))

# ------------------------------------------------------------------
# 1. Check for missing values
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: MISSING VALUES PER COLUMN")
print("=" * 70)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_report = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct.round(2)})
missing_report = missing_report[missing_report["Missing Count"] > 0]
print(missing_report if not missing_report.empty else "No missing values found.")

# Handle missing values:
# Only "Cuisines" has missing values (9 rows). Since it is a categorical
# text field, we fill missing entries with "Not Specified" rather than
# dropping rows, to avoid losing other useful information in those rows.
df["Cuisines"] = df["Cuisines"].fillna("Not Specified")

print("\nMissing values handled -> 'Cuisines' NaNs filled with 'Not Specified'.")
print("Remaining missing values:", df.isnull().sum().sum())

# ------------------------------------------------------------------
# 2. Data type conversion
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: DATA TYPES (BEFORE CONVERSION)")
print("=" * 70)
print(df.dtypes)

# Convert categorical Yes/No columns to proper category dtype
yes_no_cols = ["Has Table booking", "Has Online delivery",
                "Is delivering now", "Switch to order menu"]
for col in yes_no_cols:
    df[col] = df[col].astype("category")

# Country Code and Price range are categorical in nature even though numeric
df["Country Code"] = df["Country Code"].astype("category")
df["Price range"] = df["Price range"].astype("category")
df["Rating color"] = df["Rating color"].astype("category")
df["Rating text"] = df["Rating text"].astype("category")

print("\n" + "=" * 70)
print("STEP 3b: DATA TYPES (AFTER CONVERSION)")
print("=" * 70)
print(df.dtypes)

# ------------------------------------------------------------------
# 3. Distribution of target variable: Aggregate rating
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: DISTRIBUTION OF TARGET VARIABLE 'Aggregate rating'")
print("=" * 70)
print(df["Aggregate rating"].describe())

# Restaurants with rating 0.0 usually mean "Not rated"
zero_rating_count = (df["Aggregate rating"] == 0).sum()
print(f"\nRestaurants with Aggregate rating = 0 (Not Rated): {zero_rating_count} "
      f"({zero_rating_count/len(df)*100:.2f}% of data)")

print("\nCount of restaurants per Rating text (class imbalance check):")
print(df["Rating text"].value_counts())

# Plot distribution of the target variable
plt.figure(figsize=(9, 5))
plt.hist(df["Aggregate rating"], bins=20, color="#4C72B0", edgecolor="black")
plt.title("Distribution of Aggregate Rating")
plt.xlabel("Aggregate Rating")
plt.ylabel("Number of Restaurants")
plt.tight_layout()
plt.savefig("output_aggregate_rating_distribution.png", dpi=150)
plt.close()
print("\nSaved chart -> output_aggregate_rating_distribution.png")

# Plot class imbalance (Rating text)
plt.figure(figsize=(9, 5))
df["Rating text"].value_counts().plot(kind="bar", color="#DD8452", edgecolor="black")
plt.title("Class Distribution of Rating Text (Class Imbalance Check)")
plt.xlabel("Rating Text")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("output_rating_text_class_imbalance.png", dpi=150)
plt.close()
print("Saved chart -> output_rating_text_class_imbalance.png")

# Save cleaned dataset for reuse in subsequent tasks
df.to_csv("cleaned_dataset.csv", index=False)
print("\nCleaned dataset saved -> cleaned_dataset.csv")

print("\n" + "=" * 70)
print("SUMMARY / INSIGHTS")
print("=" * 70)
print("""
1. The dataset has 9,551 rows and 21 columns.
2. Only the 'Cuisines' column had missing values (9 rows) which were
   filled with 'Not Specified'.
3. Several numeric-looking columns (Country Code, Price range, Rating
   color, Rating text) were converted to 'category' dtype since they
   represent categorical/labelled data rather than continuous numbers.
4. The 'Aggregate rating' distribution shows a strong class imbalance:
   a large share of restaurants have a rating of 0.0 (Not Rated), while
   genuine ratings mostly cluster between 2.5 and 4.5. This imbalance
   should be considered before building any predictive model (Level 3).
""")

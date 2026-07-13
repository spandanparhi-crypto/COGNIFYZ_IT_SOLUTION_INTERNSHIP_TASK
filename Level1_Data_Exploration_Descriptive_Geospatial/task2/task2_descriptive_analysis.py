"""
Cognifyz Technologies - Data Science Internship
LEVEL 1 - TASK 2: Descriptive Analysis
------------------------------------------------------
Objectives:
1. Calculate basic statistical measures (mean, median, std, etc.) for
   numerical columns.
2. Explore the distribution of categorical variables like "Country Code",
   "City", and "Cuisines".
3. Identify the top cuisines and cities with the highest number of
   restaurants.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")
df["Cuisines"] = df["Cuisines"].fillna("Not Specified")

# ------------------------------------------------------------------
# 1. Statistical measures for numerical columns
# ------------------------------------------------------------------
numeric_cols = ["Average Cost for two", "Price range", "Aggregate rating", "Votes"]

print("=" * 70)
print("STEP 1: DESCRIPTIVE STATISTICS FOR NUMERICAL COLUMNS")
print("=" * 70)
stats = df[numeric_cols].describe().T
stats["median"] = df[numeric_cols].median()
print(stats)

# ------------------------------------------------------------------
# 2. Distribution of categorical variables
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: DISTRIBUTION OF CATEGORICAL VARIABLES")
print("=" * 70)

print(f"\nUnique Country Codes : {df['Country Code'].nunique()}")
print(df["Country Code"].value_counts().head(10))

print(f"\nUnique Cities        : {df['City'].nunique()}")
print(df["City"].value_counts().head(10))

print(f"\nUnique Cuisine combos: {df['Cuisines'].nunique()}")
print(df["Cuisines"].value_counts().head(10))

# ------------------------------------------------------------------
# 3. Top cuisines (split multi-cuisine strings) and top cities
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: TOP CUISINES AND CITIES BY NUMBER OF RESTAURANTS")
print("=" * 70)

# Split "Cuisines" on comma since a restaurant may serve multiple cuisines
all_cuisines = df["Cuisines"].str.split(",").explode().str.strip()
top_cuisines = all_cuisines.value_counts().head(10)
print("\nTop 10 individual cuisines (restaurant serves this cuisine):")
print(top_cuisines)

top_cities = df["City"].value_counts().head(10)
print("\nTop 10 cities by number of restaurants:")
print(top_cities)

# ------------------------------------------------------------------
# 4. Visualizations
# ------------------------------------------------------------------
plt.figure(figsize=(10, 5))
top_cuisines.plot(kind="bar", color="#55A868", edgecolor="black")
plt.title("Top 10 Cuisines by Number of Restaurants")
plt.xlabel("Cuisine")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("output_top_cuisines.png", dpi=150)
plt.close()
print("\nSaved chart -> output_top_cuisines.png")

plt.figure(figsize=(10, 5))
top_cities.plot(kind="bar", color="#C44E52", edgecolor="black")
plt.title("Top 10 Cities by Number of Restaurants")
plt.xlabel("City")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("output_top_cities.png", dpi=150)
plt.close()
print("Saved chart -> output_top_cities.png")

print("\n" + "=" * 70)
print("SUMMARY / INSIGHTS")
print("=" * 70)
print(f"""
1. Average cost for two ranges widely; mean = {df['Average Cost for two'].mean():.2f},
   median = {df['Average Cost for two'].median():.2f}, indicating a right-skewed
   distribution with some very expensive outliers.
2. The dataset spans {df['Country Code'].nunique()} countries and
   {df['City'].nunique()} cities, but restaurant density is heavily
   concentrated in a small number of cities (see top 10 list above).
3. The most common individual cuisine is '{top_cuisines.index[0]}',
   followed by '{top_cuisines.index[1]}' and '{top_cuisines.index[2]}'.
4. '{top_cities.index[0]}' has the highest number of listed restaurants
   in this dataset.
""")

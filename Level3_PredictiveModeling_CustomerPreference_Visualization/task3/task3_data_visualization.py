"""
Cognifyz Technologies - Data Science Internship
LEVEL 3 - TASK 3: Data Visualization
------------------------------------------------------
Objectives:
1. Create visualizations to represent the distribution of ratings using
   different charts (histogram, bar plot, etc.).
2. Compare the average ratings of different cuisines or cities using
   appropriate visualizations.
3. Visualize the relationship between various features and the target
   variable to gain insights.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")
df["Cuisines"] = df["Cuisines"].fillna("Not Specified")

# ------------------------------------------------------------------
# 1. Distribution of ratings - histogram + box plot
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df["Aggregate rating"], bins=20, color="#4C72B0", edgecolor="black")
axes[0].set_title("Histogram of Aggregate Rating")
axes[0].set_xlabel("Aggregate Rating")
axes[0].set_ylabel("Frequency")

axes[1].boxplot(df["Aggregate rating"], vert=True, patch_artist=True,
                 boxprops=dict(facecolor="#DD8452"))
axes[1].set_title("Box Plot of Aggregate Rating")
axes[1].set_ylabel("Aggregate Rating")

plt.tight_layout()
plt.savefig("output_rating_distribution_combo.png", dpi=150)
plt.close()
print("Saved chart -> output_rating_distribution_combo.png")

# ------------------------------------------------------------------
# 2. Average rating comparison - cuisines & cities
# ------------------------------------------------------------------
cuisine_df = df.copy()
cuisine_df["Cuisines"] = cuisine_df["Cuisines"].str.split(",")
cuisine_df = cuisine_df.explode("Cuisines")
cuisine_df["Cuisines"] = cuisine_df["Cuisines"].str.strip()
cuisine_counts = cuisine_df["Cuisines"].value_counts()
frequent_cuisines = cuisine_counts[cuisine_counts >= 30].index

avg_rating_cuisine = (cuisine_df[cuisine_df["Cuisines"].isin(frequent_cuisines)]
                       .groupby("Cuisines")["Aggregate rating"].mean()
                       .sort_values(ascending=False).head(10))

top_cities = df["City"].value_counts().head(10).index
avg_rating_city = (df[df["City"].isin(top_cities)]
                    .groupby("City")["Aggregate rating"].mean()
                    .sort_values(ascending=False))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

avg_rating_cuisine.plot(kind="bar", ax=axes[0], color="#55A868", edgecolor="black")
axes[0].set_title("Top 10 Cuisines by Average Rating")
axes[0].set_ylabel("Average Rating")
axes[0].tick_params(axis="x", rotation=45)

avg_rating_city.plot(kind="bar", ax=axes[1], color="#C44E52", edgecolor="black")
axes[1].set_title("Average Rating - Top 10 Busiest Cities")
axes[1].set_ylabel("Average Rating")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("output_avg_rating_cuisine_city.png", dpi=150)
plt.close()
print("Saved chart -> output_avg_rating_cuisine_city.png")

# ------------------------------------------------------------------
# 3. Relationship between features and target variable
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Votes vs Rating
axes[0, 0].scatter(df["Votes"], df["Aggregate rating"], alpha=0.3, s=10, c="#4C72B0")
axes[0, 0].set_title("Votes vs Aggregate Rating")
axes[0, 0].set_xlabel("Votes")
axes[0, 0].set_ylabel("Aggregate Rating")
axes[0, 0].set_xscale("log")

# Average Cost for two vs Rating
axes[0, 1].scatter(df["Average Cost for two"], df["Aggregate rating"],
                    alpha=0.3, s=10, c="#DD8452")
axes[0, 1].set_title("Average Cost for Two vs Aggregate Rating")
axes[0, 1].set_xlabel("Average Cost for Two")
axes[0, 1].set_ylabel("Aggregate Rating")
axes[0, 1].set_xscale("log")

# Price range vs Rating (box plot)
price_groups = [df[df["Price range"] == p]["Aggregate rating"] for p in sorted(df["Price range"].unique())]
axes[1, 0].boxplot(price_groups, labels=sorted(df["Price range"].unique()),
                    patch_artist=True, boxprops=dict(facecolor="#55A868"))
axes[1, 0].set_title("Price Range vs Aggregate Rating")
axes[1, 0].set_xlabel("Price Range")
axes[1, 0].set_ylabel("Aggregate Rating")

# Table booking vs Rating (box plot)
booking_groups = [df[df["Has Table booking"] == b]["Aggregate rating"] for b in ["Yes", "No"]]
axes[1, 1].boxplot(booking_groups, labels=["Yes", "No"],
                    patch_artist=True, boxprops=dict(facecolor="#C44E52"))
axes[1, 1].set_title("Has Table Booking vs Aggregate Rating")
axes[1, 1].set_xlabel("Has Table Booking")
axes[1, 1].set_ylabel("Aggregate Rating")

plt.tight_layout()
plt.savefig("output_feature_vs_rating_relationships.png", dpi=150)
plt.close()
print("Saved chart -> output_feature_vs_rating_relationships.png")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY / INSIGHTS")
print("=" * 70)
print("""
1. The rating histogram/box plot confirms a spike at 0.0 (unrated
   restaurants) with the remaining ratings roughly following a
   left-skewed distribution concentrated between 3.0 and 4.5.
2. Certain cuisines and cities consistently achieve higher average
   ratings than others, useful for benchmarking restaurant performance.
3. Scatter/box plots show: (a) restaurants with more votes tend to have
   higher, more stable ratings, (b) higher price range and having table
   booking both associate with modestly higher median ratings, showing
   these features carry real predictive signal for the Level 3 - Task 1
   regression models.
""")

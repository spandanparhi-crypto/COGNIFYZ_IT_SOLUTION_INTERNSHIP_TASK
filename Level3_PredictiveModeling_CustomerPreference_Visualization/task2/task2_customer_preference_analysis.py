"""
Cognifyz Technologies - Data Science Internship
LEVEL 3 - TASK 2: Customer Preference Analysis
------------------------------------------------------
Objectives:
1. Analyze the relationship between the type of cuisine and the
   restaurant's rating.
2. Identify the most popular cuisines among customers based on the
   number of votes.
3. Determine if there are any specific cuisines that tend to receive
   higher ratings.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")
df["Cuisines"] = df["Cuisines"].fillna("Not Specified")

# Explode multi-cuisine rows into one row per (restaurant, cuisine) pair
cuisine_df = df.copy()
cuisine_df["Cuisines"] = cuisine_df["Cuisines"].str.split(",")
cuisine_df = cuisine_df.explode("Cuisines")
cuisine_df["Cuisines"] = cuisine_df["Cuisines"].str.strip()

# ------------------------------------------------------------------
# 1. Relationship between cuisine type and rating
# ------------------------------------------------------------------
print("=" * 70)
print("STEP 1: AVERAGE RATING BY CUISINE (cuisines with >= 30 restaurants)")
print("=" * 70)

cuisine_counts = cuisine_df["Cuisines"].value_counts()
frequent_cuisines = cuisine_counts[cuisine_counts >= 30].index

avg_rating_by_cuisine = (cuisine_df[cuisine_df["Cuisines"].isin(frequent_cuisines)]
                          .groupby("Cuisines")["Aggregate rating"].mean()
                          .sort_values(ascending=False))
print(avg_rating_by_cuisine)

plt.figure(figsize=(10, 6))
avg_rating_by_cuisine.head(15).plot(kind="barh", color="#4C72B0", edgecolor="black")
plt.title("Top 15 Cuisines by Average Rating (min. 30 restaurants)")
plt.xlabel("Average Aggregate Rating")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("output_top_rated_cuisines.png", dpi=150)
plt.close()
print("\nSaved chart -> output_top_rated_cuisines.png")

# ------------------------------------------------------------------
# 2. Most popular cuisines based on number of votes
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: MOST POPULAR CUISINES BY TOTAL VOTES")
print("=" * 70)

votes_by_cuisine = (cuisine_df.groupby("Cuisines")["Votes"].sum()
                     .sort_values(ascending=False))
print(votes_by_cuisine.head(15))

plt.figure(figsize=(10, 6))
votes_by_cuisine.head(15).plot(kind="barh", color="#DD8452", edgecolor="black")
plt.title("Top 15 Most Popular Cuisines by Total Votes")
plt.xlabel("Total Votes")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("output_most_popular_cuisines_votes.png", dpi=150)
plt.close()
print("\nSaved chart -> output_most_popular_cuisines_votes.png")

# ------------------------------------------------------------------
# 3. Cuisines that tend to receive consistently higher ratings
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: CUISINES WITH CONSISTENTLY HIGH RATINGS")
print("=" * 70)

cuisine_stats = (cuisine_df[cuisine_df["Cuisines"].isin(frequent_cuisines)]
                  .groupby("Cuisines")["Aggregate rating"]
                  .agg(["mean", "std", "count"])
                  .sort_values("mean", ascending=False))
print(cuisine_stats.head(10))

print("\n" + "=" * 70)
print("SUMMARY / INSIGHTS")
print("=" * 70)
print(f"""
1. Among cuisines with a meaningful sample size (>=30 restaurants),
   '{avg_rating_by_cuisine.index[0]}' has the highest average rating
   ({avg_rating_by_cuisine.iloc[0]:.2f}), followed by
   '{avg_rating_by_cuisine.index[1]}' and '{avg_rating_by_cuisine.index[2]}'.
2. Based on total customer votes, '{votes_by_cuisine.index[0]}' is the
   most popular cuisine overall, indicating strong customer engagement
   even if it isn't always the highest-rated.
3. Cuisines with both a high average rating AND low standard deviation
   (see table above) represent consistently well-reviewed categories,
   making them a safer bet for customers versus cuisines with high
   variance in ratings.
""")

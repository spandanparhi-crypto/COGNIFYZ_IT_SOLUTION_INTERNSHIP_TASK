"""
Cognifyz Technologies - Data Science Internship
LEVEL 2 - TASK 2: Price Range Analysis
------------------------------------------------------
Objectives:
1. Determine the most common price range among all restaurants.
2. Calculate the average rating for each price range.
3. Identify the color that represents the highest average rating among
   different price ranges.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")
df["Cuisines"] = df["Cuisines"].fillna("Not Specified")

# ------------------------------------------------------------------
# 1. Most common price range
# ------------------------------------------------------------------
print("=" * 70)
print("STEP 1: MOST COMMON PRICE RANGE")
print("=" * 70)

price_range_counts = df["Price range"].value_counts().sort_index()
print(price_range_counts)
most_common_price_range = price_range_counts.idxmax()
print(f"\nMost common price range: {most_common_price_range} "
      f"({price_range_counts.max()} restaurants)")

plt.figure(figsize=(7, 5))
price_range_counts.plot(kind="bar", color="#4C72B0", edgecolor="black")
plt.title("Number of Restaurants per Price Range")
plt.xlabel("Price Range (1=Low ... 4=High)")
plt.ylabel("Number of Restaurants")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("output_price_range_counts.png", dpi=150)
plt.close()
print("\nSaved chart -> output_price_range_counts.png")

# ------------------------------------------------------------------
# 2. Average rating per price range
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: AVERAGE RATING PER PRICE RANGE")
print("=" * 70)

avg_rating_by_price = df.groupby("Price range")["Aggregate rating"].mean().sort_index()
print(avg_rating_by_price)

plt.figure(figsize=(7, 5))
avg_rating_by_price.plot(kind="bar", color="#55A868", edgecolor="black")
plt.title("Average Aggregate Rating per Price Range")
plt.xlabel("Price Range (1=Low ... 4=High)")
plt.ylabel("Average Aggregate Rating")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("output_avg_rating_by_price_range.png", dpi=150)
plt.close()
print("\nSaved chart -> output_avg_rating_by_price_range.png")

# ------------------------------------------------------------------
# 3. Rating color associated with the highest average rating
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: RATING COLOR FOR HIGHEST-RATED PRICE RANGE")
print("=" * 70)

best_price_range = avg_rating_by_price.idxmax()
best_avg_rating = avg_rating_by_price.max()
print(f"Price range with the highest average rating: {best_price_range} "
      f"(avg rating = {best_avg_rating:.2f})")

# The dataset itself uses a "Rating color" field as a visual scale for
# Aggregate rating (e.g. Dark Green = Excellent, down to Red = Poor,
# White = Not rated). To find which color represents the highest average
# rating, we compute the average rating for each color band, excluding
# "White" (Not rated / rating = 0, which is not a real quality band).
color_rating_map = (df[df["Rating color"] != "White"]
                     .groupby("Rating color")["Aggregate rating"].mean()
                     .sort_values(ascending=False))
print("\nAverage Aggregate rating for each Rating color band:")
print(color_rating_map)

top_color = color_rating_map.idxmax()
print(f"\n==> The color representing the highest average rating overall "
      f"is: '{top_color}' (avg rating = {color_rating_map.max():.2f})")

# Cross-check: what color band does price range {best_price_range}'s
# average rating fall closest to?
closest_color = (color_rating_map - best_avg_rating).abs().idxmin()
print(f"\nThe average rating of the top price range ({best_avg_rating:.2f}) "
      f"falls closest to the '{closest_color}' color band.")

print("\n" + "=" * 70)
print("SUMMARY / INSIGHTS")
print("=" * 70)
print(f"""
1. The most common price range is {most_common_price_range}, with
   {price_range_counts.max()} restaurants.
2. Average rating increases with price range: range 1 has the lowest
   average rating, while range {best_price_range} has the highest average
   rating ({avg_rating_by_price.max():.2f}). This suggests pricier
   restaurants tend to be rated somewhat better on average.
3. Across the platform's rating-color scale, '{top_color}' is the color
   that represents the highest average rating band overall (typically
   used for 'Excellent' reviews). The top price range's average rating
   of {best_avg_rating:.2f} falls closest to the '{closest_color}' color
   band on that scale.
""")

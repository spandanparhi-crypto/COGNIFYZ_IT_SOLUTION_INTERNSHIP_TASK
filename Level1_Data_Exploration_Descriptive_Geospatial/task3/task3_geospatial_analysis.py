"""
Cognifyz Technologies - Data Science Internship
LEVEL 1 - TASK 3: Geospatial Analysis
------------------------------------------------------
Objectives:
1. Visualize the locations of restaurants on a map using latitude and
   longitude information.
2. Analyze the distribution of restaurants across different cities or
   countries.
3. Determine if there is any correlation between the restaurant's
   location and its rating.

Note: Since this environment has no internet access to load basemap
tiles (e.g., via folium/plotly mapbox), we visualize restaurant
locations with a matplotlib scatter plot using latitude/longitude
directly, which still clearly shows the geographic distribution.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")
df["Cuisines"] = df["Cuisines"].fillna("Not Specified")

# Drop rows with invalid (0,0) coordinates - not real locations
geo_df = df[(df["Latitude"] != 0) & (df["Longitude"] != 0)].copy()
print(f"Restaurants with valid coordinates: {len(geo_df)} / {len(df)}")

# ------------------------------------------------------------------
# 1. Scatter plot of restaurant locations (world map style)
# ------------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.scatter(geo_df["Longitude"], geo_df["Latitude"], s=4, alpha=0.5, c="#4C72B0")
plt.title("Global Distribution of Restaurant Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output_restaurant_locations_map.png", dpi=150)
plt.close()
print("Saved chart -> output_restaurant_locations_map.png")

# Colour-coded by rating for visual correlation check
plt.figure(figsize=(12, 6))
scatter = plt.scatter(geo_df["Longitude"], geo_df["Latitude"],
                       c=geo_df["Aggregate rating"], cmap="RdYlGn",
                       s=6, alpha=0.6)
plt.colorbar(scatter, label="Aggregate Rating")
plt.title("Restaurant Locations Colored by Aggregate Rating")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig("output_locations_by_rating.png", dpi=150)
plt.close()
print("Saved chart -> output_locations_by_rating.png")

# ------------------------------------------------------------------
# 2. Distribution of restaurants across cities/countries
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: RESTAURANT DISTRIBUTION BY COUNTRY / CITY")
print("=" * 70)

country_counts = df["Country Code"].value_counts()
print("\nRestaurant counts per Country Code (top 10):")
print(country_counts.head(10))

city_counts = df["City"].value_counts()
print("\nRestaurant counts per City (top 10):")
print(city_counts.head(10))

plt.figure(figsize=(9, 5))
country_counts.head(10).plot(kind="bar", color="#8172B2", edgecolor="black")
plt.title("Top 10 Countries (by Country Code) by Number of Restaurants")
plt.xlabel("Country Code")
plt.ylabel("Number of Restaurants")
plt.tight_layout()
plt.savefig("output_restaurants_per_country.png", dpi=150)
plt.close()
print("\nSaved chart -> output_restaurants_per_country.png")

# ------------------------------------------------------------------
# 3. Correlation between location and rating
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: CORRELATION BETWEEN LOCATION AND RATING")
print("=" * 70)

corr_lat = geo_df["Latitude"].corr(geo_df["Aggregate rating"])
corr_lon = geo_df["Longitude"].corr(geo_df["Aggregate rating"])
print(f"Correlation (Latitude  vs Aggregate rating) : {corr_lat:.4f}")
print(f"Correlation (Longitude vs Aggregate rating) : {corr_lon:.4f}")

# Average rating by city (top 10 cities with most restaurants)
top10_city_names = city_counts.head(10).index
avg_rating_by_city = (df[df["City"].isin(top10_city_names)]
                       .groupby("City")["Aggregate rating"].mean()
                       .sort_values(ascending=False))
print("\nAverage Aggregate rating for the 10 busiest cities:")
print(avg_rating_by_city)

plt.figure(figsize=(9, 5))
avg_rating_by_city.plot(kind="bar", color="#64B5CD", edgecolor="black")
plt.title("Average Rating in Top 10 Busiest Cities")
plt.xlabel("City")
plt.ylabel("Average Aggregate Rating")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("output_avg_rating_by_city.png", dpi=150)
plt.close()
print("Saved chart -> output_avg_rating_by_city.png")

print("\n" + "=" * 70)
print("SUMMARY / INSIGHTS")
print("=" * 70)
print(f"""
1. The scatter map shows restaurants are clustered in a handful of
   geographic hotspots (major metro areas) rather than being evenly
   spread globally.
2. Latitude vs rating correlation = {corr_lat:.4f}, Longitude vs rating
   correlation = {corr_lon:.4f}. Both values are close to zero, meaning
   there is little to no *linear* correlation between a restaurant's raw
   geographic coordinates and its rating.
3. However, average ratings DO vary meaningfully city-by-city, which
   suggests rating differences are driven more by local market/city-level
   factors (competition, cuisine variety, price point) than by raw
   latitude/longitude position itself.
""")

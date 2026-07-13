"""
Cognifyz Technologies - Data Science Internship
LEVEL 2 - TASK 3: Feature Engineering
------------------------------------------------------
Objectives:
1. Extract additional features from existing columns, such as the
   length of the restaurant name or address.
2. Create new features like "Has Table Booking" or "Has Online Delivery"
   by encoding categorical variables.
"""

import pandas as pd

df = pd.read_csv("dataset.csv")
df["Cuisines"] = df["Cuisines"].fillna("Not Specified")

# ------------------------------------------------------------------
# 1. Extract length-based features
# ------------------------------------------------------------------
print("=" * 70)
print("STEP 1: EXTRACTING LENGTH-BASED FEATURES")
print("=" * 70)

df["Restaurant Name Length"] = df["Restaurant Name"].astype(str).apply(len)
df["Address Length"] = df["Address"].astype(str).apply(len)

# Bonus: number of cuisines served (also a useful engineered feature)
df["Cuisine Count"] = df["Cuisines"].astype(str).apply(lambda x: len(x.split(",")))

print(df[["Restaurant Name", "Restaurant Name Length",
          "Address", "Address Length", "Cuisines", "Cuisine Count"]].head(10))

print("\nRestaurant Name Length stats:")
print(df["Restaurant Name Length"].describe())
print("\nAddress Length stats:")
print(df["Address Length"].describe())
print("\nCuisine Count stats:")
print(df["Cuisine Count"].describe())

# ------------------------------------------------------------------
# 2. Encode categorical variables into binary features
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: ENCODING CATEGORICAL FEATURES (BINARY 0/1)")
print("=" * 70)

df["Has Table Booking (encoded)"] = (df["Has Table booking"] == "Yes").astype(int)
df["Has Online Delivery (encoded)"] = (df["Has Online delivery"] == "Yes").astype(int)
df["Is Delivering Now (encoded)"] = (df["Is delivering now"] == "Yes").astype(int)
df["Switch To Order Menu (encoded)"] = (df["Switch to order menu"] == "Yes").astype(int)

print(df[["Has Table booking", "Has Table Booking (encoded)",
          "Has Online delivery", "Has Online Delivery (encoded)"]].head(10))

# ------------------------------------------------------------------
# 3. Save the engineered dataset
# ------------------------------------------------------------------
output_cols = [
    "Restaurant ID", "Restaurant Name", "Restaurant Name Length",
    "Address", "Address Length", "Cuisines", "Cuisine Count",
    "Has Table booking", "Has Table Booking (encoded)",
    "Has Online delivery", "Has Online Delivery (encoded)",
    "Is delivering now", "Is Delivering Now (encoded)",
    "Switch to order menu", "Switch To Order Menu (encoded)",
    "Price range", "Aggregate rating"
]
df[output_cols].to_csv("feature_engineered_dataset.csv", index=False)
print("\nSaved -> feature_engineered_dataset.csv")

print("\n" + "=" * 70)
print("SUMMARY / INSIGHTS")
print("=" * 70)
print(f"""
1. Added 'Restaurant Name Length' and 'Address Length' as new numeric
   features derived from text columns (avg name length =
   {df['Restaurant Name Length'].mean():.1f} chars, avg address length =
   {df['Address Length'].mean():.1f} chars).
2. Added 'Cuisine Count' capturing how many cuisines each restaurant
   serves (avg = {df['Cuisine Count'].mean():.2f}).
3. Encoded the four Yes/No service columns (Table booking, Online
   delivery, Is delivering now, Switch to order menu) into binary 0/1
   features, which are now ready to be used directly in a machine
   learning model (see Level 3 - Task 1).
""")

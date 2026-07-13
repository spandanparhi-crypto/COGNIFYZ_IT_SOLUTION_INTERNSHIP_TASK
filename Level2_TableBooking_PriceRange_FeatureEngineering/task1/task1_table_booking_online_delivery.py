"""
Cognifyz Technologies - Data Science Internship
LEVEL 2 - TASK 1: Table Booking and Online Delivery
------------------------------------------------------
Objectives:
1. Determine the percentage of restaurants that offer table booking
   and online delivery.
2. Compare the average ratings of restaurants with table booking and
   those without.
3. Analyze the availability of online delivery among restaurants with
   different price ranges.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")
df["Cuisines"] = df["Cuisines"].fillna("Not Specified")

# ------------------------------------------------------------------
# 1. Percentage offering table booking / online delivery
# ------------------------------------------------------------------
print("=" * 70)
print("STEP 1: PERCENTAGE OF RESTAURANTS OFFERING EACH SERVICE")
print("=" * 70)

table_booking_pct = (df["Has Table booking"] == "Yes").mean() * 100
online_delivery_pct = (df["Has Online delivery"] == "Yes").mean() * 100

print(f"Restaurants offering Table Booking : {table_booking_pct:.2f}%")
print(f"Restaurants offering Online Delivery: {online_delivery_pct:.2f}%")

plt.figure(figsize=(6, 5))
plt.bar(["Table Booking", "Online Delivery"],
        [table_booking_pct, online_delivery_pct],
        color=["#4C72B0", "#DD8452"], edgecolor="black")
plt.title("Percentage of Restaurants Offering Each Service")
plt.ylabel("Percentage (%)")
for i, v in enumerate([table_booking_pct, online_delivery_pct]):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("output_service_availability_pct.png", dpi=150)
plt.close()
print("\nSaved chart -> output_service_availability_pct.png")

# ------------------------------------------------------------------
# 2. Average rating: table booking vs no table booking
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: AVERAGE RATING - TABLE BOOKING vs NO TABLE BOOKING")
print("=" * 70)

avg_rating_booking = df.groupby("Has Table booking")["Aggregate rating"].mean()
print(avg_rating_booking)

plt.figure(figsize=(6, 5))
avg_rating_booking.plot(kind="bar", color=["#C44E52", "#55A868"], edgecolor="black")
plt.title("Average Rating: Table Booking vs No Table Booking")
plt.xlabel("Has Table Booking")
plt.ylabel("Average Aggregate Rating")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("output_avg_rating_table_booking.png", dpi=150)
plt.close()
print("Saved chart -> output_avg_rating_table_booking.png")

# ------------------------------------------------------------------
# 3. Online delivery availability by price range
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: ONLINE DELIVERY AVAILABILITY BY PRICE RANGE")
print("=" * 70)

delivery_by_price = (df.groupby("Price range")["Has Online delivery"]
                      .apply(lambda x: (x == "Yes").mean() * 100))
print(delivery_by_price)

plt.figure(figsize=(7, 5))
delivery_by_price.plot(kind="bar", color="#8172B2", edgecolor="black")
plt.title("Online Delivery Availability (%) by Price Range")
plt.xlabel("Price Range (1=Low ... 4=High)")
plt.ylabel("% Offering Online Delivery")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("output_delivery_by_price_range.png", dpi=150)
plt.close()
print("\nSaved chart -> output_delivery_by_price_range.png")

print("\n" + "=" * 70)
print("SUMMARY / INSIGHTS")
print("=" * 70)
print(f"""
1. Only {table_booking_pct:.2f}% of restaurants offer table booking,
   while {online_delivery_pct:.2f}% offer online delivery.
2. Restaurants that offer table booking have a noticeably higher average
   rating ({avg_rating_booking.get('Yes', 0):.2f}) compared to those that
   do not ({avg_rating_booking.get('No', 0):.2f}). This may reflect that
   higher-end / more established restaurants tend to offer booking.
3. Online delivery availability is highest among lower/mid price-range
   restaurants (price range 1-2) and drops off for the most expensive
   restaurants (price range 4), which more often rely on dine-in/booking
   rather than delivery.
""")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, skew, kurtosis

# Load the dataset
# ⚠️ Replace with your actual file path
df = pd.read_csv("google.csv")
print(df.head())

# -------------------------
# Clean Search Volume column
# (Convert "5M+" → 5000000, "200K+" → 200000, etc.)
# -------------------------
def parse_volume(val):
    val = str(val).replace('+', '').strip()
    if 'M' in val:
        return float(val.replace('M', '')) * 1_000_000
    elif 'K' in val:
        return float(val.replace('K', '')) * 1_000
    else:
        return float(val)

df['Search Volume Numeric'] = df['Search volume'].apply(parse_volume)

# -------------------------
# Categorize Trends into Groups
# -------------------------
def categorize(trend):
    trend = str(trend).lower()
    if any(word in trend for word in ['vs', 'cricket', 'ipl', 'football', 'sport', 'match', 'test']):
        return 'Sports'
    elif any(word in trend for word in ['diwali', 'rangoli', 'festival', 'puja']):
        return 'Festival'
    elif any(word in trend for word in ['share', 'stock', 'gold', 'price', 'ipo', 'investment', 'trading']):
        return 'Finance'
    elif any(word in trend for word in ['movie', 'song', 'film', 'actor', 'singer', 'music']):
        return 'Entertainment'
    else:
        return 'Other'

df['Category'] = df['Trends'].apply(categorize)

print("\nCategory Distribution:")
print(df['Category'].value_counts())

# -------------------------
# Calculate Stats for Each Category
# -------------------------
search = df['Search Volume Numeric']

mean_val = search.mean()
median_val = search.median()
mode_val = search.mode()[0]
std_val = search.std()
skew_val = skew(search)
kurtosis_val = kurtosis(search)

print("\n--- Overall Search Volume Stats ---")
print("Mean:", mean_val)
print("Median:", median_val)
print("Mode:", mode_val)
print("Standard Deviation:", std_val)
print("Skewness:", skew_val)
print("Kurtosis:", kurtosis_val)

# -------------------------
# 1. Normal Distribution of Search Volume
# -------------------------
mean = search.mean()
std = search.std()

plt.figure(figsize=(10, 5))
plt.hist(search, bins=30, density=True, color='steelblue', alpha=0.7)

x = np.linspace(search.min(), search.max(), 1000)
y = norm.pdf(x, mean, std)
plt.plot(x, y, color='red', linewidth=2)

plt.title("Normal Distribution of Search Volume")
plt.xlabel("Search Volume")
plt.ylabel("Probability Density")
plt.tight_layout()
plt.show()

# -------------------------
# 2. Skewed Distribution
# -------------------------
plt.figure(figsize=(10, 5))
plt.hist(search, bins=30, density=True, color='orange', alpha=0.7)
plt.title("Skewed Distribution of Search Volume")
plt.xlabel("Search Volume")
plt.ylabel("Density")
plt.tight_layout()
plt.show()

skew_value = skew(search)
print("\nSkewness of Search Volume:", skew_value)

if skew_value > 0:
    print("Distribution is Positively Skewed (Right Skewed)")
elif skew_value < 0:
    print("Distribution is Negatively Skewed (Left Skewed)")
else:
    print("Distribution is Symmetric")

# -------------------------
# 3. ✅ COMPARISON: Avg Search Volume by Category
# -------------------------
category_avg = df.groupby('Category')['Search Volume Numeric'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=category_avg.index, y=category_avg.values, palette='viridis')
plt.title("Avg Search Volume by Category (Sports vs Finance vs Festival etc.)")
plt.xlabel("Category")
plt.ylabel("Average Search Volume")
plt.tight_layout()
plt.show()

# -------------------------
# 4. ✅ COMPARISON: Boxplot - Search Volume Distribution per Category
# -------------------------
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Category', y='Search Volume Numeric', palette='Set2')
plt.title("Search Volume Distribution by Category")
plt.xlabel("Category")
plt.ylabel("Search Volume")
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------
# Load the Google Trends dataset
# -----------------------------------------------
df = pd.read_csv("google.csv")
print(df.head())
print(df.columns.tolist())


# -----------------------------------------------
# 1. Identify missing values
# -----------------------------------------------

# Shows True/False for each cell
print(df.isnull())

# Count missing values per column
print(df.isnull().sum())

# Rows where 'Search volume' is missing
print(df[df["Search volume"].isnull()])

# All rows that have any missing value
print(df[df.isnull().any(axis=1)])

# Total missing values in entire DataFrame
print(df.isnull().sum().sum())


# -----------------------------------------------
# 2. Handle missing values
# -----------------------------------------------

# (A) Fill 'Search volume' missing values with Mode (most common value)
#     (It's a categorical column like "5M+", "200K+" — so Mode makes more sense than Mean)
df['Search volume'] = df['Search volume'].fillna(df['Search volume'].mode()[0])
print(df['Search volume'])

# (B) Fill 'Trends' missing values with a Custom Value
df['Trends'] = df['Trends'].fillna('Unknown')
print(df['Trends'])

# (C) Fill 'Trend breakdown' missing values with Mode
df['Trend breakdown'] = df['Trend breakdown'].fillna(df['Trend breakdown'].mode()[0])
print(df['Trend breakdown'])

# (D) Fill 'Started' and 'Ended' with a Custom Value
df['Started'] = df['Started'].fillna('Not Available')
df['Ended'] = df['Ended'].fillna('Not Available')


# -----------------------------------------------
# 3. Prepare Search Volume for numeric comparison
# -----------------------------------------------

# Convert 'Search volume' like "5M+", "200K+" → numeric values
def convert_volume(val):
    if pd.isnull(val):
        return 0
    val = str(val).replace('+', '').strip()
    if 'M' in val:
        return float(val.replace('M', '')) * 1_000_000
    elif 'K' in val:
        return float(val.replace('K', '')) * 1_000
    else:
        return float(val)

df['Search volume numeric'] = df['Search volume'].apply(convert_volume)
print(df[['Trends', 'Search volume', 'Search volume numeric']].head(10))


# -----------------------------------------------
# 4. Detect outliers in Search Volume using Box Plot
# -----------------------------------------------

plt.figure(figsize=(6, 5))
plt.boxplot(df['Search volume numeric'])
plt.title("Boxplot Before Removing Outliers (Search Volume)")
plt.ylabel("Search Volume")
plt.show()

# IQR Method on 'Search volume numeric'
Q1 = df['Search volume numeric'].quantile(0.25)
Q3 = df['Search volume numeric'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Search volume numeric'] < lower_bound) | (df['Search volume numeric'] > upper_bound)]
print("Outlier Trends:\n", outliers[['Trends', 'Search volume']])

print("Original shape:", df.shape)
print("Outliers found:", outliers.shape)


# -----------------------------------------------
# 5. COMPARISON: Top 15 Trends by Search Volume  ← MAIN COMPARISON
# -----------------------------------------------

# Remove outliers for cleaner comparison
df_clean = df[(df['Search volume numeric'] >= lower_bound) & (df['Search volume numeric'] <= upper_bound)]

# Sort and take top 15
top15 = df_clean.nlargest(15, 'Search volume numeric')

plt.figure(figsize=(12, 6))
sns.barplot(data=top15, x='Search volume numeric', y='Trends', palette='viridis')
plt.title("Top 15 Google Trends by Search Volume (Outliers Removed)")
plt.xlabel("Search Volume")
plt.ylabel("Trends")
plt.tight_layout()
plt.show()


# -----------------------------------------------
# 6. Boxplot After Removing Outliers
# -----------------------------------------------

plt.figure(figsize=(6, 5))
plt.boxplot(df_clean['Search volume numeric'])
plt.title("Boxplot After Removing Outliers (Search Volume)")
plt.ylabel("Search Volume")
plt.show()

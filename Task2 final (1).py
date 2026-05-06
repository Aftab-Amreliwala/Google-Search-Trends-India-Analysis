# Task 2 : Exploratory Data Analysis (EDA) - Google Trends Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Load the dataset
# -------------------------
df = pd.read_csv("google.csv")
print(df.head())
print("\nColumns:", df.columns.tolist())

# -------------------------
# Data Cleaning
# -------------------------
# Convert 'Search volume' from string like '5M+', '200K+', '100+' to numeric
def convert_volume(val):
    val = str(val).replace('+', '').strip()
    if 'M' in val:
        return float(val.replace('M', '')) * 1_000_000
    elif 'K' in val:
        return float(val.replace('K', '')) * 1_000
    else:
        return float(val)

df['Search_volume_num'] = df['Search volume'].apply(convert_volume)

# Convert Started and Ended to datetime
df['Started'] = pd.to_datetime(df['Started'], errors='coerce', utc=True)
df['Ended'] = pd.to_datetime(df['Ended'], errors='coerce', utc=True)

# Calculate trend duration in hours
df['Duration_hours'] = (df['Ended'] - df['Started']).dt.total_seconds() / 3600

# Extract start hour to see at what time trends usually start
df['Start_hour'] = df['Started'].dt.hour

# Extract start date
df['Start_date'] = df['Started'].dt.date

# -------------------------
# 1. Univariate Analysis - Numerical Column: Search_volume_num
# -------------------------
print("\nUnivariate Analysis of Search Volume (Numeric)")
print(df['Search_volume_num'].describe())

plt.figure()
plt.hist(df['Search_volume_num'], bins=20, color='steelblue', edgecolor='black')
plt.title("Histogram of Search Volume")
plt.xlabel("Search Volume (Numeric)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# -------------------------
# 2. Univariate Analysis - Categorical Column: Search volume (original)
# -------------------------
print("\nValue Counts of Search Volume Category:")
print(df['Search volume'].value_counts())

plt.figure(figsize=(10, 5))
df['Search volume'].value_counts().plot(kind='bar', color='coral', edgecolor='black')
plt.title("Bar Chart of Search Volume Categories")
plt.xlabel("Search Volume Category")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --------------------------------
# 3. Numerical vs Numerical
# Search Volume vs Duration Hours (Scatter Plot)
# Do higher search volume trends last longer?
# --------------------------------
plt.figure()
plt.scatter(df['Search_volume_num'], df['Duration_hours'], alpha=0.5, color='green')
plt.xlabel("Search Volume (Numeric)")
plt.ylabel("Trend Duration (Hours)")
plt.title("Search Volume vs Trend Duration")
plt.tight_layout()
plt.show()

# --------------------------------
# 4. Categorical vs Numerical
# Search Volume Category vs Duration Hours (Box Plot)
# Which category of trend lasts longer?
# --------------------------------

# Sort categories by magnitude for better readability
volume_order = ['100+', '200+', '500+', '1K+', '2K+', '5K+', '10K+',
                '20K+', '50K+', '100K+', '200K+', '500K+', '1M+', '2M+', '5M+', '10M+']
existing_order = [v for v in volume_order if v in df['Search volume'].unique()]

plt.figure(figsize=(14, 6))
sns.boxplot(data=df, x='Search volume', y='Duration_hours',
            order=existing_order, hue='Search volume', palette='Set2', legend=False)
plt.title("Trend Duration by Search Volume Category")
plt.suptitle("")
plt.xlabel("Search Volume Category")
plt.ylabel("Duration (Hours)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --------------------------------
# 5. Categorical vs Categorical
# Search Volume Category vs Start Hour group (Count Plot)
# Do high-volume trends start at certain times of the day?
# --------------------------------

# Create time-of-day buckets
def time_bucket(hour):
    if 5 <= hour < 12:
        return 'Morning (5-12)'
    elif 12 <= hour < 17:
        return 'Afternoon (12-17)'
    elif 17 <= hour < 21:
        return 'Evening (17-21)'
    else:
        return 'Night (21-5)'

df['Time_of_day'] = df['Start_hour'].apply(time_bucket)

cross_tab = pd.crosstab(df['Time_of_day'], df['Search volume'])
print("\nTime of Day vs Search Volume Category:\n", cross_tab)

cross_tab.plot(kind='bar', figsize=(10, 5))
plt.title("Time of Day vs Search Volume Category")
plt.xlabel("Time of Day")
plt.ylabel("Count")
plt.xticks(rotation=30, ha='right')
plt.legend(title='Search Volume', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --------------------------------
# 6. Correlation Matrix
# Search Volume (numeric) vs Duration Hours vs Start Hour
# --------------------------------
print("\nCorrelation Matrix:")
print(df[['Search_volume_num', 'Duration_hours', 'Start_hour']].corr())

# --------------------------------
# 7. Multivariate Pair Plots
# --------------------------------
sns.pairplot(df[['Search_volume_num', 'Duration_hours', 'Start_hour']].dropna())
plt.suptitle("Pair Plot: Search Volume, Duration, Start Hour", y=1.02)
plt.show()

# --------------------------------
# 8. Multivariate HeatMap
# --------------------------------
'''
📊 Interpretation (Very Important)
Color           Meaning
Dark Red        Strong positive correlation
Blue            Weak or negative correlation
Light           Moderate correlation
'''

num_df = df[['Search_volume_num', 'Duration_hours', 'Start_hour']].dropna()
corr = num_df.corr()

plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Heatmap: Correlation Between Search Volume, Duration & Start Hour")
plt.tight_layout()
plt.show()

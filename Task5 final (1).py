import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, skew, kurtosis

# Load the dataset using Pandas
df = pd.read_csv("google.csv")
print(df)

def full_eda(df):
    print("DATASET INFO")
    df.info()
    print("\n" + "-"*50)

    # 2️ Descriptive Statistics
    print("DESCRIPTIVE STATISTICS")
    print(df.describe(include='all'))  # include='all' because columns are strings
    print("\n" + "-"*50)

    # 3️ Check Missing Values
    print("MISSING VALUES")
    print(df.isnull().sum())
    print("\n" + "-"*50)

    # 4️ Search Volume - Convert to numeric and analyze
    print("SEARCH VOLUME ANALYSIS")

    # Clean the 'Search volume' column — remove '+', 'K', 'M' and convert
    def parse_volume(val):
        val = str(val).replace('+', '').strip()
        if 'M' in val:
            return float(val.replace('M', '')) * 1_000_000
        elif 'K' in val:
            return float(val.replace('K', '')) * 1_000
        else:
            return float(val)

    df['Search volume numeric'] = df['Search volume'].apply(parse_volume)

    # 5️ Top 10 Trending Topics by Search Volume
    print("\nTOP 10 TRENDS BY SEARCH VOLUME:")
    top10 = df[['Trends', 'Search volume numeric']].sort_values(
        by='Search volume numeric', ascending=False
    ).head(10)
    print(top10.to_string(index=False))
    print("\n" + "-"*50)

    # 6️ Comparison: Trends vs Search Volume (Bar Chart)
    print("COMPARISON: Trends vs Search Volume")
    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=top10,
        x='Search volume numeric',
        y='Trends',
        palette='viridis'
    )
    plt.title('Top 10 Google Trends by Search Volume')
    plt.xlabel('Search Volume (numeric)')
    plt.ylabel('Trend')
    plt.tight_layout()
    plt.savefig('trends_vs_search_volume.png')
    plt.show()
    print("Chart saved as trends_vs_search_volume.png")

full_eda(df)
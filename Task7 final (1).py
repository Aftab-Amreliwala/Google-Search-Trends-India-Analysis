# Task 7: Supervised Learning – Regression Model

# Split Dataset into Training & Testing Data

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Load dataset
df = pd.read_csv("google.csv")

# ── Feature Engineering ──────────────────────────────────────────────────────
# Convert 'Search volume' (e.g. "5M+", "200K+") → numeric values
def parse_volume(v):
    v = str(v).strip().replace('+', '')
    if 'M' in v:
        return float(v.replace('M', '')) * 1_000_000
    elif 'K' in v:
        return float(v.replace('K', '')) * 1_000
    else:
        return float(v)

df['volume_numeric'] = df['Search volume'].apply(parse_volume)

# Use length of the Trend name as a simple numeric feature
df['trend_name_length'] = df['Trends'].apply(len)

# Also use number of words in the Trend name
df['trend_word_count'] = df['Trends'].apply(lambda x: len(x.split()))


# ── Simple Linear Regression ─────────────────────────────────────────────────
# Predict: Search Volume  |  Feature: Trend Name Length

X = df[['trend_name_length']]
y = df['volume_numeric']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

slr = LinearRegression()
slr.fit(X_train, y_train)

y_pred = slr.predict(X_test)

print("── Simple Linear Regression ──")
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))


# ── Multiple Linear Regression ───────────────────────────────────────────────
# Predict: Search Volume  |  Features: Trend Name Length + Word Count

X_multi = df[['trend_name_length', 'trend_word_count']]
y = df['volume_numeric']

X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

mlr = LinearRegression()
mlr.fit(X_train, y_train)

y_pred = mlr.predict(X_test)

print("\n── Multiple Linear Regression ──")
print("R2 Score (MLR):", r2_score(y_test, y_pred))
print("MSE (MLR):", mean_squared_error(y_test, y_pred))


# ── Logistic Regression ───────────────────────────────────────────────────────
# Classify: High Volume (≥ 500K) vs Low Volume (< 500K)

df['high_volume'] = (df['volume_numeric'] >= 500_000).astype(int)

X = df[['trend_name_length', 'trend_word_count']]
y = df['high_volume']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)

print("\n── Logistic Regression ──")
print("Accuracy:", accuracy_score(y_test, y_pred))
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset
df = pd.read_csv("google.csv")

# --- Data Preprocessing ---

# Convert Search Volume (e.g., "500K+", "2M+") to numeric values
def parse_volume(val):
    val = str(val).replace('+', '').strip()
    if 'M' in val:
        return float(val.replace('M', '')) * 1_000_000
    elif 'K' in val:
        return float(val.replace('K', '')) * 1_000
    else:
        return float(val)

df['Search Volume Numeric'] = df['Search volume'].apply(parse_volume)

# Convert Started and Ended to datetime, then calculate Trend Duration (in hours)
df['Started'] = pd.to_datetime(df['Started'], utc=True)
df['Ended']   = pd.to_datetime(df['Ended'],   utc=True)
df['Trend Duration (hrs)'] = (df['Ended'] - df['Started']).dt.total_seconds() / 3600

# Drop rows with invalid/negative durations
df = df[df['Trend Duration (hrs)'] > 0]

# --- Regression ---

# Independent Variable: Search Volume
# Dependent Variable: Trend Duration (hours)
X = df[['Search Volume Numeric']]
y = df['Trend Duration (hrs)']

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Print model parameters
print("Intercept:", model.intercept_)
print("Slope (Coefficient):", model.coef_[0])
print(f"\nFormula: Trend Duration = ({model.coef_[0]:.8f} × Search Volume) + {model.intercept_:.2f}")

# Plot the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.4, color='steelblue', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel("Search Volume (Numeric)")
plt.ylabel("Trend Duration (Hours)")
plt.title("Linear Regression: Does Higher Search Volume = Longer Trend?")
plt.legend()
plt.tight_layout()
plt.show()
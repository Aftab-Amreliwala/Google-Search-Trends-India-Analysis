import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("google.csv")

# -----------------------------------------
# Feature Engineering
# -----------------------------------------

# 1. Convert Search volume to numeric
#    (e.g., "5M+" → 5000000, "500K+" → 500000)
def parse_volume(vol):
    vol = str(vol).replace('+', '').strip()
    if 'M' in vol:
        return float(vol.replace('M', '')) * 1_000_000
    elif 'K' in vol:
        return float(vol.replace('K', '')) * 1_000
    else:
        return float(vol)

df['volume_numeric'] = df['Search volume'].apply(parse_volume)

# 2. Convert Started and Ended to datetime
df['Started'] = pd.to_datetime(df['Started'], errors='coerce', utc=True)
df['Ended']   = pd.to_datetime(df['Ended'],   errors='coerce', utc=True)

# 3. Calculate trend duration in hours
df['duration_hours'] = (df['Ended'] - df['Started']).dt.total_seconds() / 3600

# Drop rows with missing values
df = df.dropna(subset=['duration_hours', 'volume_numeric'])

# -----------------------------------------
# Classification Target
# -----------------------------------------
# High Volume = 1 (Search volume >= 500K)
# Low Volume  = 0 (Search volume <  500K)
df['high_volume'] = df['volume_numeric'].apply(lambda x: 1 if x >= 500_000 else 0)

# -----------------------------------------
# Features and Target
# -----------------------------------------
X = df[['volume_numeric', 'duration_hours']]
y = df['high_volume']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# -----------------------------------------
# Confusion Matrix
# -----------------------------------------
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Volume", "High Volume"])
disp.plot()
plt.title("Confusion Matrix - Google Trends")
plt.show()

'''
[[TN  FP]
 [FN  TP]]

TN (True Negative):  Correctly predicted Low Volume
FP (False Positive): Predicted High Volume but actually Low Volume
FN (False Negative): Predicted Low Volume but actually High Volume
TP (True Positive):  Correctly predicted High Volume
'''
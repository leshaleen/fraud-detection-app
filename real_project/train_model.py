import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset (ONLY needed columns)
df = pd.read_csv(
    "data/PS_20174392719_1491204439457_log.csv",
    usecols=[
        "step", "type", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "isFraud"
    ],
    nrows=50000   # 🔥 ONLY 50K ROWS (FAST)
)

# Encode 'type' manually (NO get_dummies explosion)
df["type"] = df["type"].astype("category").cat.codes

# Split
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
model = RandomForestClassifier(n_estimators=10)
model.fit(X_scaled, y)

# Save
joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ MODEL CREATED SUCCESSFULLY")
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("ml/data/creditcard.csv")

# Simulate additional columns
df["country"] = ["US"] * len(df)  # Adjust as per your simulation logic
df["merchant_id"] = ["m1"] * len(df)

# Load encoders and scaler
encoder_country = joblib.load("ml/models/encoder_country.pkl")
encoder_merchant = joblib.load("ml/models/encoder_merchant.pkl")
scaler = joblib.load("ml/models/scaler.pkl")
model = joblib.load("ml/models/fraud_model.pkl")

# Encode categorical features
df["country"] = encoder_country.transform(df["country"])
df["merchant_id"] = encoder_merchant.transform(df["merchant_id"])

# Select known fraud samples (Class == 1)
frauds = df[df["Class"] == 1]

if not frauds.empty:
    print(f"Found {len(frauds)} fraud samples.")
    # Test the first fraud sample
    sample = frauds.iloc[0]

    # Prepare features for prediction
    features = (
        ["Time"] +
        [f"V{i}" for i in range(1, 29)] +
        ["Amount", "country", "merchant_id"]
    )

    X_sample = sample[features].values.reshape(1, -1)
    X_sample_scaled = scaler.transform(X_sample)

    # Predict fraud score
    score = model.predict_proba(X_sample_scaled)[0][1]
    print(f"Fraud prediction score: {score:.4f}")
    print("Fraud detected!" if score > 0.5 else "No fraud detected.")
else:
    print("No fraud samples found.")

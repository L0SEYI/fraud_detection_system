import joblib
import numpy as np
from app.models.request import TransactionInput

# Load trained artifacts
model = joblib.load("ml/models/fraud_model.pkl")
scaler = joblib.load("ml/models/scaler.pkl")
encoder_country = joblib.load("ml/models/encoder_country.pkl")
encoder_merchant = joblib.load("ml/models/encoder_merchant.pkl")

def predict_transaction(data: TransactionInput):
    data_dict = data.dict()

    # Extract and order features
    features = [
        data_dict["Time"],
        *[data_dict[f"V{i}"] for i in range(1, 29)],
        data_dict["Amount"]
    ]

    # Log raw features
    print("[DEBUG] Raw numeric features:", features)

    # Validate & encode categorical fields
    if data_dict["country"] not in encoder_country.classes_:
        print(f"[WARN] Unknown country: {data_dict['country']}")
    if data_dict["merchant_id"] not in encoder_merchant.classes_:
        print(f"[WARN] Unknown merchant: {data_dict['merchant_id']}")

    country_encoded = encoder_country.transform([data_dict["country"]])[0]
    merchant_encoded = encoder_merchant.transform([data_dict["merchant_id"]])[0]

    # Append encoded fields
    features.append(country_encoded)
    features.append(merchant_encoded)

    # Prepare input array
    input_array = np.array(features).reshape(1, -1)

    # Scale input
    scaled_input = scaler.transform(input_array)
    print("[DEBUG] Scaled input:", scaled_input.tolist())

    # Predict
    score = model.predict_proba(scaled_input)[0][1]
    print("[DEBUG] Prediction score:", score)

    return {
        "is_fraud": bool(score > 0.5),
        "score": round(float(score), 4)
    }

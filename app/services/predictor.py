import joblib
import numpy as np
from app.models.request import TransactionInput

# Load trained artifacts
model = joblib.load("ml/models/fraud_model.pkl")
scaler = joblib.load("ml/models/scaler.pkl")
encoder_country = joblib.load("ml/models/encoder_country.pkl")
encoder_merchant = joblib.load("ml/models/encoder_merchant.pkl")

def predict_transaction(data: TransactionInput):
    # Convert input to dict and extract values
    data_dict = data.dict()

    # Extract raw features
    features = [
        data_dict[f"V{i}"] for i in range(1, 29)
    ]
    features.insert(0, data_dict["Time"])  # Time at position 0
    features.append(data_dict["Amount"])   # Amount at the end

    # Encode categorical features
    country_encoded = encoder_country.transform([data_dict["country"]])[0]
    merchant_encoded = encoder_merchant.transform([data_dict["merchant_id"]])[0]

    # Append encoded features
    features.append(country_encoded)
    features.append(merchant_encoded)

    # Convert to NumPy and reshape
    input_array = np.array(features).reshape(1, -1)

    # Scale input (uses the same order as training)
    scaled_input = scaler.transform(input_array)

    # Predict
    score = model.predict_proba(scaled_input)[0][1]
    return {
        "is_fraud": score > 0.5,
        "score": round(score, 4)
    }

import joblib
import numpy as np

model = joblib.load("ml/models/fraud_model.pkl")
scaler = joblib.load("ml/models/scaler.pkl")

def predict_transaction(data):
    input_data = np.array([list(data.dict().values())])
    scaled = scaler.transform(input_data)
    score = model.predict_proba(scaled)[0][1]
    return {"is_fraud": score > 0.5, "score": score}

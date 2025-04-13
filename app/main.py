from fastapi import FastAPI
from app.models.request import TransactionInput
from app.services.predictor import predict_transaction
from app.routers import auth
from app.auth import get_current_user


app = FastAPI(
    title="Fraud Detection API",
    description="API for real-time fraud prediction using machine learning",
    version="1.0.0"
)

@app.post("/predict", tags=["Fraud Detection"])
async def predict(data: TransactionInput):
    prediction = predict_transaction(data)
    return prediction

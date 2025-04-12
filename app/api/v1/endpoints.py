from fastapi import APIRouter
from app.models.request import TransactionInput
from app.models.response import FraudPrediction
from app.services.predictor import predict_transaction

router = APIRouter()

@router.post("/predict", response_model=FraudPrediction)
async def predict(data: TransactionInput):
    result = predict_transaction(data)
    return {"is_fraud": result["is_fraud"], "score": result["score"]}

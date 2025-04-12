from pydantic import BaseModel

class FraudPrediction(BaseModel):
    is_fraud: bool
    score: float

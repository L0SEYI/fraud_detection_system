from fastapi import FastAPI, Depends, HTTPException
from app.models.request import TransactionInput
from app.services.predictor import predict_transaction
from app.auth.auth_handler import create_access_token
from app.auth.users import authenticate_user
from pydantic import BaseModel

class LoginData(BaseModel):
    username: str
    password: str


app = FastAPI(
    title="Fraud Detection API",
    description="API for real-time fraud prediction using machine learning",
    version="1.0.0"
)

@app.post("/predict", tags=["Fraud Detection"])
async def predict(data: TransactionInput):
    prediction = predict_transaction(data)
    return prediction



@app.post("/login")
def login(data: LoginData):
    user = authenticate_user(data.username, data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}
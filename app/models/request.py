from pydantic import BaseModel

class TransactionInput(BaseModel):
    Time: float
    V1: float
    V2: float
    ...
    Amount: float

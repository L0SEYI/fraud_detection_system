def rules_based_check(data):
    return (
        data.Amount > 1000 and 
        data.V1 < -1.5 and 
        data.V3 > 2.0
    )
def predict_transaction(data):
    if rules_based_check(data):
        return {"is_fraud": True, "score": 1.0}
    ...

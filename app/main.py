from fastapi import FastAPI
from app.api.v1.endpoints import router

app = FastAPI()
app.include_router(router, prefix="/api/v1")

# app/auth/auth_bearer.py
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .auth_handler import decode_token

class JWTBearer(HTTPBearer):
    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        if credentials and credentials.scheme == "Bearer":
            try:
                payload = decode_token(credentials.credentials)
                return payload
            except:
                raise HTTPException(status_code=403, detail="Invalid token or expired")
        raise HTTPException(status_code=403, detail="Invalid auth scheme")

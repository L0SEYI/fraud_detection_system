# app/db/mock_db.py

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory user database
fake_users_db = {
    "john_doe": {
        "username": "john_doe",
        "full_name": "John Doe",
        "email": "john@example.com",
        "hashed_password": pwd_context.hash("secret123"),
        "disabled": False,
    },
    "jane_smith": {
        "username": "jane_smith",
        "full_name": "Jane Smith",
        "email": "jane@example.com",
        "hashed_password": pwd_context.hash("password456"),
        "disabled": False,
    },
}

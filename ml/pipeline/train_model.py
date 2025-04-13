import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os

# Load dataset
df = pd.read_csv("ml/data/creditcard.csv")

# Simulate additional fields for demo
# In real data, you should have these columns already
df["country"] = np.random.choice(["US", "UK", "CA", "DE"], size=len(df))
df["merchant_id"] = np.random.choice(["m1", "m2", "m3"], size=len(df))

# Split target and features
X = df.drop("Class", axis=1)
y = df["Class"]

# Label encode country and merchant_id
le_country = LabelEncoder()
le_merchant = LabelEncoder()

X["country_encoded"] = le_country.fit_transform(X["country"])
X["merchant_encoded"] = le_merchant.fit_transform(X["merchant_id"])

X = X.drop(["country", "merchant_id"], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

# Train model
model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss")
model.fit(X_res, y_res)

# Evaluate
y_pred = model.predict(X_test_scaled)
proba = model.predict_proba(X_test_scaled)[:, 1]

print("ROC AUC Score:", roc_auc_score(y_test, proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save models
os.makedirs("ml/models", exist_ok=True)
joblib.dump(model, "ml/models/fraud_model.pkl")
joblib.dump(scaler, "ml/models/scaler.pkl")
joblib.dump(le_country, "ml/models/encoder_country.pkl")
joblib.dump(le_merchant, "ml/models/encoder_merchant.pkl")

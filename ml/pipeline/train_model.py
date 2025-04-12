import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import os

# Load dataset
df = pd.read_csv("ml/data/creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Oversampling
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)

# Train
model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
model.fit(X_res, y_res)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save
os.makedirs("ml/models", exist_ok=True)
joblib.dump(model, "ml/models/fraud_model.pkl")
joblib.dump(scaler, "ml/models/scaler.pkl")

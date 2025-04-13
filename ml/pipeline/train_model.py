import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # adjust as needed



# Paths
DATA_PATH = "ml/data/creditcard.csv"
MODEL_DIR = "ml/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Optional: Simulate categorical columns
df["country"] = np.random.choice(["US", "UK", "CA", "IN"], size=len(df))
df["merchant_id"] = np.random.choice(["m1", "m2", "m3", "m4"], size=len(df))

# Define features & label
features = (
    ["Time"] +
    [f"V{i}" for i in range(1, 29)] +
    ["Amount", "country", "merchant_id"]
)
target = "Class"

X = df[features]
y = df[target]

# Encode categorical features
encoder_country = LabelEncoder()
encoder_merchant = LabelEncoder()

X.loc[:, "merchant_id"] = encoder_merchant.fit_transform(X["merchant_id"])
X.loc[:, "country"] = encoder_country.fit_transform(X["country"])

# Save encoders
joblib.dump(encoder_country, f"{MODEL_DIR}/encoder_country.pkl")
joblib.dump(encoder_merchant, f"{MODEL_DIR}/encoder_merchant.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

# Oversample using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

# Train model with class imbalance handling
scale_pos_weight = len(y_res[y_res == 0]) / len(y_res[y_res == 1])
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_res, y_res)

# Predict & Evaluate
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("✅ ROC AUC:", round(roc_auc_score(y_test, y_proba), 4))
print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, f"{MODEL_DIR}/fraud_model.pkl")

print("✅ Model training complete and artifacts saved.")



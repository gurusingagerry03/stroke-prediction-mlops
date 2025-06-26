import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import mlflow
import mlflow.sklearn
from imblearn.over_sampling import SMOTE
import joblib
import os

# === Load and prepare dataset ===
df = pd.read_csv("stroke-data.csv")
df = df.drop("id", axis=1)
df["bmi"] = df["bmi"].fillna(df["bmi"].median())

# Encode categorical features
label_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Split features and target
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# === Apply SMOTE to balance the training set ===
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# === Start MLflow experiment tracking ===
with mlflow.start_run():
    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # === Calculate evaluation metrics ===
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # === Log parameters and metrics to MLflow ===
    mlflow.log_param("model_type", "RandomForestClassifier + SMOTE")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")

    # === Output to terminal ===
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

# === Save model to disk ===
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

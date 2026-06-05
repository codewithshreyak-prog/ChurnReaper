import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from preprocessing import load_and_preprocess_data

# Load data
X_train, X_test, y_train, y_test, feature_names = (
    load_and_preprocess_data(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
)

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Probabilities
y_prob = model.predict_proba(X_test)[:, 1]

print("\n========== RESULTS ==========")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nROC AUC:")
print(roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/churn_model.pkl")

print("\nModel saved successfully!")
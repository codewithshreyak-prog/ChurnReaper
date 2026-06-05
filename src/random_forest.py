import joblib

from sklearn.ensemble import RandomForestClassifier
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

# Create Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# Train
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print("\n========== RANDOM FOREST RESULTS ==========\n")

print("Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nROC-AUC:")
print(roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(rf, "models/random_forest.pkl")

print("\nRandom Forest saved successfully!")
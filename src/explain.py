import os
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from preprocessing import load_and_preprocess_data

print("Loading data...")

# Load data
X_train, X_test, y_train, y_test, feature_names = (
    load_and_preprocess_data(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
)

print("Training Random Forest...")

# Train model
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

print("Generating SHAP values...")

# Create SHAP explainer
explainer = shap.TreeExplainer(rf)

# Calculate SHAP values
shap_values = explainer(X_test)

print("X_test shape:", X_test.shape)
print("SHAP shape:", shap_values.values.shape)

# Create screenshots folder if missing
os.makedirs("screenshots", exist_ok=True)

# Generate SHAP summary plot
plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values.values[:, :, 1],  # Churn class
    X_test,
    show=False
)

plt.tight_layout()

# Save image
plt.savefig(
    "screenshots/shap_summary.png",
    bbox_inches="tight"
)

print("\nSHAP summary saved to:")
print("screenshots/shap_summary.png")

# Display plot
plt.show()
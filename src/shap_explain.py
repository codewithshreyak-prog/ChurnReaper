import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from preprocessing import load_and_preprocess_data

X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

explainer = shap.TreeExplainer(rf)

shap_values = explainer(X_test)

print("X_test shape:", X_test.shape)
print("SHAP values shape:", shap_values.values.shape)

# For binary classification, use class 1 = churn
shap.summary_plot(
    shap_values.values[:, :, 1],
    X_test,
    show=True
)
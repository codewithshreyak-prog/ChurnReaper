import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from preprocessing import load_and_preprocess_data

# Load data
X_train, X_test, y_train, y_test, feature_names = (
    load_and_preprocess_data(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

# Feature importance
importance = rf.feature_importances_

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Most Important Features:\n")
print(feature_df.head(10))

# Plot
top10 = feature_df.head(10)

plt.figure(figsize=(10,6))
plt.barh(
    top10["Feature"],
    top10["Importance"]
)

plt.title("Top 10 Churn Drivers")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.show()
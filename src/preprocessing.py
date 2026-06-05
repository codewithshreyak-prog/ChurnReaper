import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Remove customerID because it does not help prediction
    df = df.drop("customerID", axis=1)

    # Convert TotalCharges from text to number
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges values
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Convert target column: Yes/No → 1/0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Separate input features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Convert text columns into numeric columns
    X = pd.get_dummies(X, drop_first=True)

    # Split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, X.columns


if __name__ == "__main__":
    file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(file_path)

    print("Preprocessing completed successfully!")
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    print("Number of features:", len(feature_names))
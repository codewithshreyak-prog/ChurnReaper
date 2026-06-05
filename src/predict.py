import joblib
import pandas as pd
from src.preprocessing import load_and_preprocess_data


MODEL_PATH = "models/churn_model.pkl"
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def predict_churn(customer_data):
    model = joblib.load(MODEL_PATH)

    _, _, _, _, feature_names = load_and_preprocess_data(DATA_PATH)

    input_df = pd.DataFrame([customer_data])

    input_encoded = pd.get_dummies(input_df)

    input_encoded = input_encoded.reindex(
        columns=feature_names,
        fill_value=0
    )

    churn_probability = model.predict_proba(input_encoded)[0][1]
    prediction = model.predict(input_encoded)[0]

    return prediction, churn_probability


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 475.0
    }

    prediction, probability = predict_churn(sample_customer)

    print("Prediction:", "Churn" if prediction == 1 else "No Churn")
    print("Churn Probability:", round(probability * 100, 2), "%")
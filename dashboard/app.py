import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px

from src.predict import predict_churn

print("Starting ChurnReaper Dashboard...")

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

total_customers = len(df)
total_churn = len(df[df["Churn"] == "Yes"])
churn_rate = round((total_churn / total_customers) * 100, 2)

avg_monthly_charges = df["MonthlyCharges"].mean()
revenue_at_risk = round(total_churn * avg_monthly_charges, 2)

fig_churn = px.histogram(
    df,
    x="Churn",
    color="Churn",
    title="Customer Churn Distribution"
)

fig_monthly = px.box(
    df,
    x="Churn",
    y="MonthlyCharges",
    title="Monthly Charges vs Churn"
)

app = Dash(__name__)

card_style = {
    "width": "22%",
    "display": "inline-block",
    "textAlign": "center",
    "border": "1px solid #ddd",
    "borderRadius": "10px",
    "padding": "20px",
    "margin": "10px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.1)"
}

app.layout = html.Div([

    html.H1("📊 ChurnReaper Dashboard", style={"textAlign": "center"}),

    html.P(
        "Predict customer churn, identify key churn drivers, and estimate revenue at risk.",
        style={"textAlign": "center", "fontSize": "18px"}
    ),

    html.Br(),

    html.Div([
        html.Div([html.H3("Total Customers"), html.H2(str(total_customers))], style=card_style),
        html.Div([html.H3("Customers Churned"), html.H2(str(total_churn))], style=card_style),
        html.Div([html.H3("Churn Rate"), html.H2(f"{churn_rate}%")], style=card_style),
        html.Div([html.H3("Revenue at Risk"), html.H2(f"${revenue_at_risk:,.2f}")], style=card_style),
    ]),

    html.Hr(),

    dcc.Graph(figure=fig_churn),

    html.Br(),

    dcc.Graph(figure=fig_monthly),

    html.Hr(),

    html.H2("Top Churn Drivers", style={"textAlign": "center"}),

    html.Img(
        src=app.get_asset_url("feature_importance.png"),
        style={
            "width": "80%",
            "display": "block",
            "margin": "auto"
        }
    ),

    html.Hr(),

    html.H2("Model Explainability", style={"textAlign": "center"}),

    html.Img(
        src=app.get_asset_url("shap_summary.png"),
        style={
            "width": "90%",
            "display": "block",
            "margin": "auto"
        }
    ),

    html.Hr(),

    html.H2("🔮 Predict Customer Churn"),

    html.Label("Tenure"),
    dcc.Input(id="tenure", type="number", value=5, style={"width": "100%", "padding": "8px"}),

    html.Br(),
    html.Br(),

    html.Label("Monthly Charges"),
    dcc.Input(id="monthly-charges", type="number", value=95, style={"width": "100%", "padding": "8px"}),

    html.Br(),
    html.Br(),

    html.Label("Contract"),
    dcc.Dropdown(
        id="contract",
        options=[
            {"label": "Month-to-month", "value": "Month-to-month"},
            {"label": "One year", "value": "One year"},
            {"label": "Two year", "value": "Two year"},
        ],
        value="Month-to-month"
    ),

    html.Br(),

    html.Label("Internet Service"),
    dcc.Dropdown(
        id="internet-service",
        options=[
            {"label": "DSL", "value": "DSL"},
            {"label": "Fiber optic", "value": "Fiber optic"},
            {"label": "No", "value": "No"},
        ],
        value="Fiber optic"
    ),

    html.Br(),

    html.Label("Payment Method"),
    dcc.Dropdown(
        id="payment-method",
        options=[
            {"label": "Electronic check", "value": "Electronic check"},
            {"label": "Mailed check", "value": "Mailed check"},
            {"label": "Bank transfer", "value": "Bank transfer (automatic)"},
            {"label": "Credit card", "value": "Credit card (automatic)"},
        ],
        value="Electronic check"
    ),

    html.Br(),

    html.Button(
        "Predict Churn",
        id="predict-button",
        n_clicks=0,
        style={
            "padding": "10px 20px",
            "fontSize": "16px",
            "cursor": "pointer"
        }
    ),

    html.H3(id="prediction-output"),

], style={"padding": "30px", "fontFamily": "Arial"})


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("tenure", "value"),
    State("monthly-charges", "value"),
    State("contract", "value"),
    State("internet-service", "value"),
    State("payment-method", "value")
)
def predict_customer(n_clicks, tenure, monthly_charges, contract, internet_service, payment_method):
    if n_clicks == 0:
        return ""

    total_charges = tenure * monthly_charges

    customer_data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet_service,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    prediction, probability = predict_churn(customer_data)
    label = "Churn" if prediction == 1 else "No Churn"

    return f"Prediction: {label} | Churn Probability: {round(probability * 100, 2)}%"


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8051)
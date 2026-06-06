from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI(
    title="Customer Churn Prediction API"
)

# =========================
# Load Model Once
# =========================

artifact = joblib.load("model.pkl")

model = artifact["model"]
feature_columns = artifact["feature_columns"]

# =========================
# Request Schema
# =========================

class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal["Yes", "No"]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0)

    PhoneService: Literal["Yes", "No"]

    MultipleLines: Literal[
        "Yes",
        "No",
        "No phone service"
    ]

    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No"
    ]

    OnlineSecurity: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    OnlineBackup: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    DeviceProtection: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    TechSupport: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    StreamingTV: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    StreamingMovies: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year"
    ]

    PaperlessBilling: Literal[
        "Yes",
        "No"
    ]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    MonthlyCharges: float = Field(..., ge=0)


@app.get("/")
def home():
    return {"message": "Customer Churn API Running"}


@app.post("/predict")
def predict(data: CustomerData):

    try:

        df = pd.DataFrame([data.dict()])
        df["SeniorCitizen"] = df["SeniorCitizen"].map({
            "No": 0,
            "Yes": 1
        })
        df["TotalCharges"] = (
            df["tenure"] *
            df["MonthlyCharges"]
        )

        # =========================
        # Feature Engineering
        # =========================


        df["AvgMonthlySpend"] = (
            df["TotalCharges"] /
            (df["tenure"] + 1)
        )

        df["IsLongTermCustomer"] = (
            df["tenure"] > 24
        ).astype(int)

        service_cols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies"
        ]

        for col in service_cols:
            df[col] = df[col].replace({
                "Yes": 1,
                "No": 0,
                "No internet service": 0
            })

        df["ServiceCount"] = df[service_cols].sum(axis=1)

        # =========================
        # One Hot Encoding
        # =========================

        df = pd.get_dummies(
            df,
            drop_first=True
        )

        # =========================
        # Match Training Columns
        # =========================

        df = df.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # prediction = model.predict(df)[0]

        # confidence = float(
        #     model.predict_proba(df)[0][1]
        # )
        prediction = int(model.predict(df)[0])
        probs = model.predict_proba(df)[0]
        confidence = float(probs[prediction])

        return {
            "prediction": int(prediction),
            "confidence": round(confidence, 4),
            "result":
                "Likely to Churn"
                if prediction == 1
                else "Likely to Stay"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
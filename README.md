# Customer Churn Prediction System

## Overview

This project predicts whether a telecommunications customer is likely to churn (leave the service) based on customer demographics, account information, subscribed services, contract details, and billing information.

The solution consists of:

* A machine learning model trained on the Telco Customer Churn dataset
* A FastAPI backend for inference
* A Streamlit web application for user interaction
* SQLite persistence for storing prediction history
* Logging for monitoring requests and errors

---

# Project Structure

```text
project/
│
├── train_model.ipynb
├── model.pkl
├── app.py
├── database.py
├── streamlit_app.py
├── predictions.db
├── app.log
├── requirements.txt
└── README.md
```

---

# Environment Setup

## 1. Create a Virtual Environment

```bash
python -m venv venv
```

## 2. Activate the Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the FastAPI Backend

Start the API server:

```bash
uvicorn app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## Swagger Documentation

```text
http://127.0.0.1:8000/docs
```

---

# Available Endpoints

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

---

## Prediction Endpoint

```http
POST /predict
```

Example Request:

```json
{
  "gender": "Male",
  "SeniorCitizen": "No",
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 24,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5
}
```

Example Response:

```json
{
  "prediction": "Churn",
  "confidence": 0.8421,
  "result": "Likely to Churn"
}
```

---

## Prediction History

```http
GET /predictions
```

Returns all stored predictions from the SQLite database.

---

# Running the Streamlit Application

Start Streamlit:

```bash
streamlit run streamlit_app.py
```

The application will open automatically in your browser.

---

# Features

## Backend

* FastAPI-based inference API
* Input validation using Pydantic
* SQLite prediction persistence
* Request and error logging
* Model loaded once at application startup
* Confidence score generation

## Frontend

* User-friendly Streamlit interface
* Dropdown-based categorical inputs
* Numeric input validation
* Prediction confidence display
* Prediction history dashboard
* Error handling for API failures

---

# Database Persistence

All predictions are stored in SQLite (`predictions.db`).

Each record contains:

* Timestamp
* Input features
* Prediction result
* Confidence score

If database writing fails, the prediction is still returned to the user and the error is logged.

---

# Logging

Application logs are stored in:

```text
app.log
```

Logged events include:

* Service startup
* Prediction requests
* Validation errors
* Database errors
* Runtime exceptions

---

# Design Notes

## Problem and Dataset

The objective of this project is to predict customer churn for a telecommunications company. Customer churn prediction is an important business problem because retaining existing customers is generally more cost-effective than acquiring new ones.

The project uses the Telco Customer Churn dataset, which contains customer demographic information, account details, subscribed services, contract information, and billing-related attributes. The target variable indicates whether a customer has churned or remained with the company.

---

## Key Design Decisions and Feature Engineering

Several machine learning models were evaluated, including Logistic Regression and Random Forest. Hyperparameter tuning was performed using GridSearchCV, and Random Forest was selected as the final model due to its strong predictive performance and ability to capture non-linear relationships.

Three engineered features were added:

### 1. AvgMonthlySpend

```text
TotalCharges / (tenure + 1)
```

This feature captures a customer's average spending behavior over their relationship with the company.

### 2. IsLongTermCustomer

```text
tenure > 24
```

A binary indicator identifying customers who have remained with the company for more than two years.

### 3. ServiceCount

Counts the number of subscribed value-added services:

* Online Security
* Online Backup
* Device Protection
* Technical Support
* Streaming TV
* Streaming Movies

This feature represents overall customer engagement with the company's services.

The model is loaded once at application startup to improve inference performance and avoid repeated disk access. SQLite was selected as the persistence layer because it is lightweight, free, and requires no external database server.

---

## Future Improvements

Given additional time, several improvements could be implemented:

* Evaluate advanced models such as XGBoost and LightGBM
* Add model explainability using SHAP values
* Improve feature engineering through additional interaction features
* Containerize the application using Docker
* Deploy the solution to a cloud platform
* Add monitoring dashboards and usage analytics
* Implement authentication and user management for production use

---

# Requirements

```text
fastapi
uvicorn
pandas
numpy
scikit-learn
streamlit
requests
joblib
pydantic
```

import requests
import json

# Test data matching the new schema
test_data = {
    "age": 35,
    "occupation_status": "Employed",
    "years_employed": 8.5,
    "annual_income": 75000,
    "credit_score": 720,
    "credit_history_years": 12.0,
    "savings_assets": 25000,
    "current_debt": 15000,
    "defaults_on_file": 0,
    "delinquencies_last_2yrs": 1,
    "derogatory_marks": 0,
    "product_type": "Personal",
    "loan_intent": "debt_consolidation",
    "loan_amount": 50000,
    "debt_to_income_ratio": 0.35
}

try:
    response = requests.get(
        "http://ec2-13-62-110-20.eu-north-1.compute.amazonaws.com:8005/health",
        timeout=30
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
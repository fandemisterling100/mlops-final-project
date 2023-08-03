"""Module to test prediction endpoint"""

import requests

data = {
    "total_outcome_dollar_amount": [2109.19],
    "total_income_dollar_amount": [1305.06],
    "risk_pld": ["HIGH"],
}

URL = "http://localhost:9696/predict"
response = requests.post(URL, json=data, timeout=10)
print(response.json())

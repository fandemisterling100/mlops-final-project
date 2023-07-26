import requests

data = {
    "total_outcome_dollar_amount": [2109.19],
    "total_income_dollar_amount": [1305.06],
    "risk_pld": ["HIGH"],
}

url = "http://localhost:9696/predict"
response = requests.post(url, json=data)
print(response.json())

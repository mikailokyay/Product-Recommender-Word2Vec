import requests

url = 'http://127.0.0.1:5000/product_recommendation'
r = requests.post(url, json={
    "task": "product recommentation system",
    "product_id": "32322582",
    "product_count": 15
})

print(r.json)

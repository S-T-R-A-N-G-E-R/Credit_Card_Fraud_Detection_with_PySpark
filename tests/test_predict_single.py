# tests/test_predict_single.py
def test_predict_single(client):
    payload = {
        "TransactionID": 1234567,
        "TransactionDT": 11111111,
        "TransactionAmt": 200.5,
        "card1": 1500,
        "ProductCD": "W",
        "DeviceType": "mobile",
        "DeviceInfo": "iOS",
        "C1": 1.0,
        "C2": 2.0,
        "C3": 3.0,
        "C4": 4.0,
        "C5": 5.0,
        "C6": 6.0,
        "C7": 7.0,
        "C8": 8.0,
        "C9": 9.0,
        "C10": 10.0
    }
    response = client.post("/predict_single", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "fraud" in data
    assert isinstance(data["fraud_probability"], float)
    assert isinstance(data["fraud"], bool)

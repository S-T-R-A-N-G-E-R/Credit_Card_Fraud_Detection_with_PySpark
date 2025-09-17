# tests/test_predict_batch.py
import pandas as pd
import tempfile

def test_predict_batch(client):
    # Create dummy dataframe
    df = pd.DataFrame([
        {
            "TransactionID": 1,
            "TransactionDT": 111111,
            "TransactionAmt": 100.0,
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
            "C10": 10.0,
        }
    ])

    # Save temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp.flush()
        with open(tmp.name, "rb") as f:
            response = client.post(
                "/predict_batch",
                files={"file": ("test.csv", f, "text/csv")}
            )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert "fraud_probability" in data[0]
    assert "fraud" in data[0]

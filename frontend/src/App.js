import React, { useState } from "react";
import axios from "axios";

function App() {
  const [transaction, setTransaction] = useState({ TransactionID: 123456, TransactionAmt: 100.0 });
  const [singleResult, setSingleResult] = useState(null);

  const [batchFile, setBatchFile] = useState(null);
  const [batchResult, setBatchResult] = useState([]);

  // Handle single prediction
  const handleSingleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post("http://127.0.0.1:8000/predict_single", transaction);
      setSingleResult(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  // Handle batch prediction
  const handleBatchSubmit = async (e) => {
    e.preventDefault();
    if (!batchFile) return;

    const formData = new FormData();
    formData.append("file", batchFile);

    try {
      const res = await axios.post("http://127.0.0.1:8000/predict_batch", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setBatchResult(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div style={{ margin: "20px" }}>
      <h2>💳 Credit Card Fraud Detection</h2>

      {/* Single Transaction Form */}
      <form onSubmit={handleSingleSubmit}>
        <h3>Single Transaction</h3>
        <label>TransactionID: </label>
        <input
          type="number"
          value={transaction.TransactionID}
          onChange={(e) => setTransaction({ ...transaction, TransactionID: e.target.value })}
        />
        <br />
        <label>TransactionAmt: </label>
        <input
          type="number"
          value={transaction.TransactionAmt}
          onChange={(e) => setTransaction({ ...transaction, TransactionAmt: e.target.value })}
        />
        <br />
        <button type="submit">Predict</button>
      </form>

      {singleResult && (
        <div>
          <h4>Prediction Result</h4>
          <p>Fraud Probability: {singleResult.fraud_probability.toFixed(4)}</p>
          <p>Fraud: {singleResult.fraud ? "⚠️ YES" : "✅ NO"}</p>
        </div>
      )}

      <hr />

      {/* Batch Upload */}
      <form onSubmit={handleBatchSubmit}>
        <h3>Batch Prediction (CSV Upload)</h3>
        <input type="file" onChange={(e) => setBatchFile(e.target.files[0])} />
        <button type="submit">Upload & Predict</button>
      </form>

      {batchResult.length > 0 && (
        <div>
          <h4>Batch Results</h4>
          <table border="1" cellPadding="5">
            <thead>
              <tr>
                <th>TransactionID</th>
                <th>Fraud Probability</th>
                <th>Fraud</th>
              </tr>
            </thead>
            <tbody>
              {batchResult.map((row, i) => (
                <tr key={i}>
                  <td>{row.TransactionID}</td>
                  <td>{row.fraud_probability.toFixed(4)}</td>
                  <td>{row.fraud ? "⚠️ YES" : "✅ NO"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default App;

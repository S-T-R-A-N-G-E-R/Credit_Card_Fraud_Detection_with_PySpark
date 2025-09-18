import React, { useState } from "react";
import api from "../api";
import { Button, Box, Typography } from "@mui/material";

export default function PredictBatch() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);

  const handleSubmit = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await api.post("/predict_batch", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResults(response.data);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6">Predict Batch Transactions</Typography>
      <input
        type="file"
        accept=".csv"
        onChange={(e) => setFile(e.target.files[0])}
        style={{ marginTop: "1rem" }}
      />
      <Button variant="contained" onClick={handleSubmit} sx={{ mt: 2 }}>
        Upload & Predict
      </Button>

      {results.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle1">Results:</Typography>
          <ul>
            {results.map((r, idx) => (
              <li key={idx}>
                ID: {r.TransactionID}, FraudProb: {r.fraud_probability.toFixed(4)}, Fraud:{" "}
                {r.fraud ? "Yes 🚨" : "No ✅"}
              </li>
            ))}
          </ul>
        </Box>
      )}
    </Box>
  );
}

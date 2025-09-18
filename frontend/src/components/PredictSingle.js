import React, { useState } from "react";
import api from "../api";
import { TextField, Button, Box, Typography } from "@mui/material";

export default function PredictSingle() {
  const [transactionID, setTransactionID] = useState("");
  const [transactionAmt, setTransactionAmt] = useState("");
  const [result, setResult] = useState(null);

  const handleSubmit = async () => {
    try {
      const payload = {
        TransactionID: parseInt(transactionID) || 0,
        TransactionAmt: parseFloat(transactionAmt) || 0.0,
        // backend expects many features, but missing ones default via pipeline
      };
      const response = await api.post("/predict_single", payload);
      setResult(response.data);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h6">Predict Single Transaction</Typography>
      <TextField
        label="Transaction ID"
        value={transactionID}
        onChange={(e) => setTransactionID(e.target.value)}
        sx={{ mr: 2, mt: 2 }}
      />
      <TextField
        label="Transaction Amount"
        value={transactionAmt}
        onChange={(e) => setTransactionAmt(e.target.value)}
        sx={{ mr: 2, mt: 2 }}
      />
      <Button variant="contained" onClick={handleSubmit} sx={{ mt: 2 }}>
        Predict
      </Button>

      {result && (
        <Box sx={{ mt: 3 }}>
          <Typography>Fraud Probability: {result.fraud_probability.toFixed(4)}</Typography>
          <Typography>Fraud: {result.fraud ? "Yes 🚨" : "No ✅"}</Typography>
        </Box>
      )}
    </Box>
  );
}

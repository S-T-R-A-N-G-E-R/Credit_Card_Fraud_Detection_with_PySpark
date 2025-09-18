import React from "react";
import PredictSingle from "./components/PredictSingle";
import PredictBatch from "./components/PredictBatch";
import { Container, Typography } from "@mui/material";

function App() {
  return (
    <Container sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>
        💳 Credit Card Fraud Detection
      </Typography>
      <PredictSingle />
      <PredictBatch />
    </Container>
  );
}

export default App;

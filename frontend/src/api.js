import axios from "axios";

// point to FastAPI container (adjust if running with docker-compose)
const api = axios.create({
  baseURL: "http://127.0.0.1:8000", // or http://backend:8000 in docker-compose
});

export default api;

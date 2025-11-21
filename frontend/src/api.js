// frontend/src/api.js
import axios from "axios";
import { API_BASE } from "./config";

const API = axios.create({
  baseURL: API_BASE || window.location.origin,
  headers: {
    'Content-Type': 'application/json'
  }
});

API.interceptors.request.use((req) => {
  const token = localStorage.getItem("token");
  if (token) {
    req.headers.Authorization = `Bearer ${token}`;
  }
  return req;
});

API.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error("API Error:", error);
    return Promise.reject(error);
  }
);

export default API;
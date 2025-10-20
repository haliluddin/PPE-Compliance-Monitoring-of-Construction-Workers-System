// frontend/src/api.js
import axios from "axios";
import { API_BASE } from "./config";

const API = axios.create({
  baseURL: API_BASE || undefined, // undefined means "relative" requests go to same origin
});

API.interceptors.request.use((req) => {
  const token = localStorage.getItem("token");
  if (token) {
    req.headers.Authorization = `Bearer ${token}`;
  }
  return req;
});

export default API;
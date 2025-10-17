// frontend/src/config.js
export const API_BASE =
  (typeof window !== "undefined" && window.__ENV && window.__ENV.API_BASE) ||
  (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_API_BASE) ||
  "https://27j8aeaqzsqbky-5173.proxy.runpod.net";
  //"http://127.0.0.1:9000";

export const WS_BASE = (typeof window !== "undefined" && window.__ENV && window.__ENV.WS_URL) ||
  (API_BASE.replace(/^http/, "ws"));

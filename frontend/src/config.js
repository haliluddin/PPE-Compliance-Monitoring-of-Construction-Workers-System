// frontend/src/config.js
const runtimeEnv = (typeof window !== "undefined" && window.__ENV) ? window.__ENV : {};
const envApi = runtimeEnv.API_BASE || runtimeEnv.VITE_API_BASE || runtimeEnv.VITE_API_URL;
const envWs = runtimeEnv.WS_URL || runtimeEnv.VITE_WS_URL || runtimeEnv.WS;

// Prefer runtime env, otherwise use current origin (same scheme + host as page)
export const API_BASE = envApi || (typeof window !== "undefined" ? window.location.origin : "");

// Derive WS_BASE from runtime or API_BASE. Ensure we return a scheme that matches the page:
function _deriveWsBase() {
  if (envWs) return envWs.replace(/\/+$/, "");
  if (!API_BASE) return "";
  // if API_BASE is https://host[:port] => wss://host[:port]
  return API_BASE.replace(/^http/, "ws").replace(/\/+$/, "");
}

export const WS_BASE = _deriveWsBase();

// For debugging at runtime
if (typeof window !== "undefined") {
  // small console-help for debugging
  console.info("[config] API_BASE =", API_BASE, "WS_BASE =", WS_BASE, "window.__ENV =", runtimeEnv);
}

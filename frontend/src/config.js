const runtimeEnv = (typeof window !== "undefined" && window.__ENV) ? window.__ENV : {};
const buildEnv = typeof import.meta !== "undefined" ? import.meta.env : {};
const envApi = runtimeEnv.API_BASE || buildEnv.VITE_API_BASE || buildEnv.VITE_API_URL || "";
export const API_BASE = envApi || (typeof window !== "undefined" ? window.location.origin : "");
function _deriveWsBase() {
  const envWs = runtimeEnv.WS_URL || buildEnv.VITE_WS_URL || buildEnv.VITE_WS || "";
  if (envWs) return envWs.replace(/\/+$/, "");
  if (!API_BASE) return "";
  return API_BASE.replace(/^http/, "ws").replace(/\/+$/, "");
}
export const WS_BASE = _deriveWsBase();
if (typeof window !== "undefined") {
  console.info("[config] API_BASE =", API_BASE, "WS_BASE =", WS_BASE, "window.__ENV =", runtimeEnv);
}

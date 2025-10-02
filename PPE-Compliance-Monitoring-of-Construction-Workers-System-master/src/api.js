export const testBackend = async () => {
  try {
    const res = await fetch("http://127.0.0.1:8000/"); // must match FastAPI URL
    return await res.json();
  } catch (err) {
    console.error("Backend connection failed:", err);
    return { error: "Connection failed" };
  }
};

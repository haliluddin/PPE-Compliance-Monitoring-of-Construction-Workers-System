// frontend/src/pages/Login.jsx
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { API_BASE } from "../config";   // <-- add/remove this line as needed

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      //const response = await fetch("http://127.0.0.1:8000/login", {
      const response = await fetch(`${API_BASE}/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      let data = {};
      try {
        data = await response.json();
      } catch {
        setError("Server returned no data.");
        setLoading(false);
        return;
      }

      if (!response.ok) {
        setError(data.detail || "Invalid credentials");
        setLoading(false);
        return;
      }

      localStorage.setItem("token", data.access_token);

      if (data.user) {
        localStorage.setItem("user", JSON.stringify(data.user));
      } else {
        localStorage.setItem("user", JSON.stringify({ name: "User", email }));
      }

      navigate("/camera");

    } catch (err) {
      console.error(err);
      setError("Network/server error.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#1E1F21] p-4">
      <div className="w-full max-w-md">
        <div className="bg-[#2A2B30] rounded-xl shadow-2xl overflow-hidden border border-gray-700">
          <div className="px-8 py-6 border-b border-gray-700">
            <h2 className="text-2xl font-bold text-white text-center">Welcome</h2>
            <p className="text-gray-400 text-sm text-center mt-1">Sign in to your account</p>
          </div>

          <form onSubmit={handleSubmit} className="p-6 space-y-6">
            {error && <p className="text-red-500 text-sm text-center">{error}</p>}

            <div className="space-y-1">
              <label className="block text-sm font-medium text-gray-300">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                placeholder="admin@example.com"
                className="w-full px-4 py-2.5 bg-[#1E1F21] border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-[#5388DF] focus:border-transparent transition-all"
              />
            </div>

            <div className="space-y-1">
              <div className="flex justify-between items-center">
                <label className="block text-sm font-medium text-gray-300">Password</label>
                <button type="button" className="text-xs text-[#5388DF] hover:underline">
                  Forgot password?
                </button>
              </div>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                placeholder="••••••••"
                className="w-full px-4 py-2.5 bg-[#1E1F21] border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-[#5388DF] focus:border-transparent transition-all"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className={`w-full py-2.5 px-4 bg-[#5388DF] text-white font-medium rounded-lg hover:bg-[#3a6fc5] transition-colors ${
                loading ? "opacity-50 cursor-not-allowed" : ""
              }`}
            >
              {loading ? "Signing in..." : "Sign In"}
            </button>

            <div className="text-center text-sm text-gray-400">
              Don't have an account?{" "}
              <a
                href="/register"
                onClick={(e) => {
                  e.preventDefault();
                  navigate("/register");
                }}
                className="text-[#5388DF] hover:underline"
              >
                Sign up
              </a>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

import { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Email:", email);
    console.log("Password:", password);

    // Temporary redirect to Camera page
    navigate("/camera");
  };

  return (
    <div
      className="relative min-h-screen flex items-center justify-center bg-cover bg-center"
      style={{
        backgroundImage:
          "url('https://images.unsplash.com/photo-1535379453347-1ffd615e2e08?auto=format&fit=crop&w=1470&q=80')",
      }}
    >
      {/* Dark overlay */}
      <div className="absolute inset-0 bg-white"></div>

      <div className="relative z-10 w-full max-w-md p-8 mx-2
                      bg-white/10 backdrop-blur-md rounded-2xl
                      shadow-2xl border border-white/20">
        <h1 className="text-3xl font-bold textblue text-center mb-6">
          WELCOME
        </h1>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Email */}
          <div>
            <label className="block text-sm font-medium text-blue mb-1">
              Email
            </label>
            <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="w-full px-4 py-2 rounded-lg bg-white/40
                            text-[#19325C] placeholder-gray-300
                            border border-[#21005D]
                            focus:outline-none focus:ring-2
                            focus:ring-[#21005D] transition"
                placeholder="you@example.com"
                />

          </div>

          {/* Password */}
          <div>
            <label className="block text-sm font-medium textblue mb-1">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full px-4 py-2 rounded-lg bg-white/40
              text-[#19325C] placeholder-gray-300
              border border-[#21005D]
              focus:outline-none focus:ring-2
              focus:ring-[#21005D] transition"
              placeholder="••••••••"
            />
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            className="w-full py-2 rounded-lg bg-blue text-white
                       font-semibold text-lg
                       hover:bg-blue-700 hover:scale-[1.02]
                       transition-all duration-200"
          >
            Sign In
          </button>
        </form>

        <p className="text-center textblue mt-6 text-sm">
          Forgot your password?{" "}
          <a href="#" className="text-blue-400 hover:underline">
            Reset here
          </a>
        </p>
      </div>
    </div>
  );
}

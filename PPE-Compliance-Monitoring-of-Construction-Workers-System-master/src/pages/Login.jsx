import { useState } from "react";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Login attempt with:", { email, password });
    
    // Simple validation
  if (email && password) {
    // Only proceed if this is a direct form submission
    const isFormSubmission = e.target.tagName === 'FORM';
    if (isFormSubmission) {
      window.location.href = '/camera';
    }
  }
};

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#1E1F21] p-4">
      <div className="w-full max-w-md">
        <div className="bg-[#2A2B30] rounded-xl shadow-2xl overflow-hidden border border-gray-700">
          <div className="px-8 py-6 border-b border-gray-700">
            <h2 className="text-2xl font-bold text-white text-center">
              Welcome
            </h2>
            <p className="text-gray-400 text-sm text-center mt-1">
              Sign in to your account
            </p>
          </div>

          <form onSubmit={handleSubmit} className="p-6 space-y-6">
            <div className="space-y-1">
              <label className="block text-sm font-medium text-gray-300">
                Email
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="w-full px-4 py-2.5 bg-[#1E1F21] border border-gray-700 rounded-lg 
                          text-white placeholder-gray-500 focus:outline-none 
                          focus:ring-2 focus:ring-[#5388DF] focus:border-transparent transition-all"
                placeholder="admin@example.com"
              />
            </div>

            <div className="space-y-1">
              <div className="flex justify-between items-center">
                <label className="block text-sm font-medium text-gray-300">
                  Password
                </label>
                <button
                  type="button"
                  className="text-xs text-[#5388DF] hover:underline"
                >
                  Forgot password?
                </button>
              </div>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="w-full px-4 py-2.5 bg-[#1E1F21] border border-gray-700 rounded-lg 
                          text-white placeholder-gray-500 focus:outline-none 
                          focus:ring-2 focus:ring-[#5388DF] focus:border-transparent transition-all"
                placeholder="••••••••"
              />
            </div>

            <button
              type="submit"
              className="w-full py-2.5 px-4 bg-[#5388DF] text-white font-medium 
                        rounded-lg hover:bg-[#3a6fc5] transition-colors"
            >
              Sign In
            </button>

            <div className="text-center text-sm text-gray-400">
              Don't have an account?{" "}
              <a 
                href="/register" 
                onClick={(e) => {
                  e.preventDefault();
                  window.location.href = '/register';
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
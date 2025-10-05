import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

export default function Register() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: ""
  });
  const [errors, setErrors] = useState({});
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: "" }));
    }
  };

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.name.trim()) {
      newErrors.name = "Name is required";
    }
    
    if (!formData.email) {
      newErrors.email = "Email is required";
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = "Please enter a valid email";
    }
    
    if (!formData.password) {
      newErrors.password = "Password is required";
    } else if (formData.password.length < 6) {
      newErrors.password = "Password must be at least 6 characters";
    }
    
    if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = "Passwords do not match";
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
  e.preventDefault();
  if (!validateForm()) return;

  try {
    const response = await fetch("http://127.0.0.1:8000/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: formData.name,
        email: formData.email,
        password: formData.password
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      setErrors({ general: data.detail || "Registration failed" });
      return;
    }

    alert("Registration successful!");
    window.location.href = "/login";

  } catch (err) {
    console.error("Error:", err);
    setErrors({ general: "Server error. Please try again later." });
  }
};


  return (
    <div className="min-h-screen flex items-center justify-center bg-[#1E1F21] p-4">
    {errors.general && <p className="text-red-500 text-sm text-center">{errors.general}</p>}

      <div className="w-full max-w-md">
        <div className="bg-[#2A2B30] rounded-xl shadow-2xl overflow-hidden border border-gray-700">
          <div className="px-8 py-6 border-b border-gray-700">
            <h2 className="text-2xl font-bold text-white text-center">
              Create Account
            </h2>
            <p className="text-gray-400 text-sm text-center mt-1">
              Get started with us today
            </p>
          </div>

          <form onSubmit={handleSubmit} className="p-6 space-y-4">
            <div className="space-y-1">
              <label className="block text-sm font-medium text-gray-300">Full Name</label>
              <input
                type="text"
                name="name"
                value={formData.name}
                onChange={handleChange}
                className={`w-full px-4 py-2.5 bg-[#1E1F21] border ${
                  errors.name ? 'border-red-500' : 'border-gray-700'
                } rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-[#5388DF] focus:border-transparent transition-all`}
                placeholder="John Doe"
              />
              {errors.name && <p className="text-red-400 text-xs mt-1">{errors.name}</p>}
            </div>

            <div className="space-y-1">
              <label className="block text-sm font-medium text-gray-300">Email</label>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                className={`w-full px-4 py-2.5 bg-[#1E1F21] border ${
                  errors.email ? 'border-red-500' : 'border-gray-700'
                } rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-[#5388DF] focus:border-transparent transition-all`}
                placeholder="name@company.com"
              />
              {errors.email && <p className="text-red-400 text-xs mt-1">{errors.email}</p>}
            </div>

            <div className="space-y-1">
              <label className="block text-sm font-medium text-gray-300">Password</label>
              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                className={`w-full px-4 py-2.5 bg-[#1E1F21] border ${
                  errors.password ? 'border-red-500' : 'border-gray-700'
                } rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-[#5388DF] focus:border-transparent transition-all`}
                placeholder="••••••••"
              />
              {errors.password && <p className="text-red-400 text-xs mt-1">{errors.password}</p>}
            </div>

            <div className="space-y-1">
              <label className="block text-sm font-medium text-gray-300">Confirm Password</label>
              <input
                type="password"
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleChange}
                className={`w-full px-4 py-2.5 bg-[#1E1F21] border ${
                  errors.confirmPassword ? 'border-red-500' : 'border-gray-700'
                } rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-[#5388DF] focus:border-transparent transition-all`}
                placeholder="••••••••"
              />
              {errors.confirmPassword && <p className="text-red-400 text-xs mt-1">{errors.confirmPassword}</p>}
            </div>

            <button
              type="submit"
              className="w-full py-2.5 px-4 bg-[#5388DF] text-white font-medium rounded-lg hover:bg-[#3a6fc5] transition-colors mt-4"
            >
              Create Account
            </button>

            <div className="text-center text-sm text-gray-400 mt-4">
            Already have an account?{" "}
            <a 
                href="/login" 
                onClick={(e) => {
                e.preventDefault();
                // Clear any stored form state
                window.history.replaceState({}, document.title, "/login");
                // Force a hard navigation
                window.location.href = '/login';
                }}
                className="text-[#5388DF] hover:underline"
            >
                Sign in
            </a>
            </div>
                    </form>
        </div>
      </div>
    </div>
  );
}
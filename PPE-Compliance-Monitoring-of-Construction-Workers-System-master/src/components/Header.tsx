import React, { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

export default function Header() {
  const { pathname } = useLocation();
  const [user, setUser] = useState({ name: "", email: "" });

  // Map each route to a title
 const titles: Record<string, string> = {
  "/camera": "Camera Monitoring",
  "/notifications": "Notifications",
  "/incidents": "Incident Records",
  "/workers": "Workers Profiling",
  "/reports": "Reports",
};


  const title = titles[pathname] || "";

  // Get user info from localStorage on mount
  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
  }, []);

  return (
    <header className="sticky top-0 z-10 bg-[#1E1F23] text-white px-6 py-4 shadow-md flex items-center justify-between border-b border-gray-700">
      <h1 className="text-2xl font-bold">{title}</h1>

      {/* User Info Box */}
      {user?.email && (
        <div className="flex flex-col items-end bg-[#2A2B30] border border-gray-600 rounded-lg px-4 py-2 text-sm">
          <span className="font-semibold text-white">{user.name || "User"}</span>
          <span className="text-gray-400 text-xs">{user.email}</span>
        </div>
      )}
    </header>
  );
}

import React from "react";
import { useLocation } from "react-router-dom";

export default function Header() {
  const { pathname } = useLocation();

  // Map each route to a title
  const titles: Record<string, string> = {
    "/camera": "Camera Monitoring",
    "/notifications": "Notifications",
    "/incidents": "Incident Records",
    "/workers": "Workers Profiling",
    "/reports": "Reports",
  };

  // Fallback if path not found
  const title = titles[pathname] || "";

  return (
     <header className="sticky top-0 z-10 bg-[#1E1F23] text-white px-6 py-4 shadow-md flex items-center justify-between border-b border-gray-700">
      <h1 className="text-2xl font-bold">{title}</h1>
    </header>
  );
}

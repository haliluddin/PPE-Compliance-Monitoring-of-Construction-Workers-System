import { useLocation } from "react-router-dom";

export default function Header() {
  const { pathname } = useLocation();

  // Map each route to a title
  const titles: Record<string, string> = {
    "/camera": "Camera",
    "/notifications": "Notifications",
    "/incidents": "Incidents",
    "/workers": "Workers",
    "/reports": "Reports",
  };

  // Fallback if path not found
  const title = titles[pathname] || "";

  return (
    <header className="sticky top-0 z-10 bg-white text-blue px-6 py-4 shadow-md flex items-center justify-between">
      <h1 className="text-2xl font-bold">{title}</h1>
    </header>
  );
}

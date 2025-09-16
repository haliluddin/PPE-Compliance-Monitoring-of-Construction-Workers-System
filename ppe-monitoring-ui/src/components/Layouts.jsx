import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";

export default function Layout() {
  return (
    <div className="flex">
      <Sidebar />
      <div className="flex-1 bg-white min-h-screen">
        <Outlet /> {/* All pages render here */}
      </div>
    </div>
  );
}

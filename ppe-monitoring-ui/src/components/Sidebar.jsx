import { useState } from "react";
import { NavLink } from "react-router-dom";
import { 
  FiCamera, 
  FiBell, 
  FiAlertTriangle, 
  FiUsers, 
  FiFileText, 
  FiMenu 
} from "react-icons/fi";

export default function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const menuItems = [
    { name: "Camera", path: "/camera", icon: <FiCamera size={20} /> },
    { name: "Notifications", path: "/notifications", icon: <FiBell size={20} /> },
    { name: "Incidents", path: "/incidents", icon: <FiAlertTriangle size={20} /> },
    { name: "Workers", path: "/workers", icon: <FiUsers size={20} /> },
    { name: "Reports", path: "/reports", icon: <FiFileText size={20} /> },
  ];

  return (
    <div className={`flex flex-col bg-blue text-white h-screen transition-all duration-300 ${isCollapsed ? "w-20" : "w-64"}`}>
      {/* Hamburger menu button */}
      <div className="flex justify-end p-4">
        <button onClick={() => setIsCollapsed(!isCollapsed)} className="text-white">
          <FiMenu size={24} />
        </button>
      </div>

      {/* Menu items */}
      <nav className="flex-1 mt-4">
        {menuItems.map((item) => (
          <NavLink
            key={item.name}
            to={item.path}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-3 text-lg hover:bg-[#518ef0] transition rounded-lg m-2 ${
                isActive ? "bg-[#5388DF]" : ""
              }`
            }
          >
            {item.icon}
            {!isCollapsed && <span>{item.name}</span>}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}

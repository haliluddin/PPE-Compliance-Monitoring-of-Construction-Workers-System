import { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { 
  FiCamera, FiBell, FiAlertTriangle, 
  FiUsers, FiFileText, FiMenu, FiLogOut
} from "react-icons/fi";

export default function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false);
   const navigate = useNavigate();

  const handleLogout = () => {
    // Add any logout logic here (clear localStorage, etc)
    navigate('/');
  };


  const menuItems = [
    { name: "Camera", path: "/camera", icon: <FiCamera size={20} /> },
    { name: "Notifications", path: "/notifications", icon: <FiBell size={20} /> },
    { name: "Incidents", path: "/incidents", icon: <FiAlertTriangle size={20} /> },
    { name: "Workers", path: "/workers", icon: <FiUsers size={20} /> },
    { name: "Reports", path: "/reports", icon: <FiFileText size={20} /> },
  ];

  return (
    <div
      className={`flex flex-col sticky top-0 z-40  text-white h-screen transition-all duration-300
      ${isCollapsed ? "w-20" : "w-64"}
      bg-[#19325C]
      relative`}
    >
      <div className="absolute inset-0 backdrop-blur-sm bg-black/40 pointer-events-none"></div>

      <div className="relative z-10 flex flex-col h-full">
        {/* Hamburger menu button */}
        <div className="flex items-center justify-between p-4">
          {!isCollapsed && <h1 className="text-2xl font-bold text-white">SafetySite</h1>}
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="text-white"
          >
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
                `flex items-center gap-3 px-4 py-3 text-lg hover:bg-white/20
                 transition rounded-lg m-2 ${isActive ? "bg-white/40" : ""}`
              }
            >
              {item.icon}
              {!isCollapsed && <span>{item.name}</span>}
            </NavLink>
          ))}
        </nav>

         {/* Logout Button */}
        <button
          onClick={handleLogout}
          className="flex items-center gap-3 px-4 py-3 text-lg hover:bg-white/20
                    transition rounded-lg m-2 text-white mb-4"
        >
          <FiLogOut size={20} />
          {!isCollapsed && <span>Logout</span>}
        </button>


      </div>
    </div>
  );
}

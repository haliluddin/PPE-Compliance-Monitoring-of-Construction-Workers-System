// frontend/src/components/Sidebar.jsx
import { useEffect, useState } from "react";
import API from "../api";
import { NavLink, useNavigate } from "react-router-dom";
import { FiCamera, FiBell, FiAlertTriangle, FiUsers, FiFileText, FiMenu, FiLogOut } from "react-icons/fi";
import { useUnread } from "../context/UnreadContext";
import { WS_BASE } from "../config";

export default function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const navigate = useNavigate();
  const { unreadCount, setUnreadCount } = useUnread();

  // Fetch initial unread count
  useEffect(() => {
    API.get("/notifications").then((res) => {
      const unread = res.data.filter((n) => !n.is_read).length;
      setUnreadCount(unread);
    });
  }, []);

  // WebSocket connection
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) return;
    
    const wsBaseClean = (WS_BASE || window.location.origin.replace(/^http/, "ws")).replace(/\/+$/, "");
    const wsUrl = `${wsBaseClean}/ws/notifications?token=${encodeURIComponent(token)}`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setUnreadCount((prev) => prev + 1); // increment unread count
    };

    ws.onclose = () => console.log("❌ Notification WS disconnected");

    return () => ws.close();
  }, []);

  const handleLogout = () => navigate("/");

  const menuItems = [
    { name: "Camera", path: "/camera", icon: <FiCamera size={20} /> },
    { name: "Notifications", path: "/notifications", icon: <FiBell size={20} /> },
    { name: "Incidents", path: "/incidents", icon: <FiAlertTriangle size={20} /> },
    { name: "Workers", path: "/workers", icon: <FiUsers size={20} /> },
    { name: "Reports", path: "/reports", icon: <FiFileText size={20} /> },
  ];

  return (
    <div className={`flex flex-col sticky top-0 z-40 text-white h-screen transition-all duration-300
      ${isCollapsed ? "w-20" : "w-64"} bg-[#19325C] relative`}
    >
      <div className="absolute inset-0 backdrop-blur-sm bg-black/40 pointer-events-none"></div>
      <div className="relative z-10 flex flex-col h-full">
        <div className="flex items-center justify-between p-4">
          {!isCollapsed && <h1 className="text-2xl font-bold text-white">SafetySite</h1>}
          <button onClick={() => setIsCollapsed(!isCollapsed)} className="text-white">
            <FiMenu size={24} />
          </button>
        </div>

        <nav className="flex-1 mt-4">
          {menuItems.map((item) => (
            <NavLink
              key={item.name}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 text-lg hover:bg-white/20 transition rounded-lg m-2 ${isActive ? "bg-white/40" : ""}`
              }
            >
              <div className="relative">
                {item.icon}
                {item.name === "Notifications" && unreadCount > 0 && (
                  <span className="absolute top-0 right-0 w-3 h-3 bg-red-500 rounded-full border-2 border-[#19325C]" />
                )}
              </div>
              {!isCollapsed && <span>{item.name}</span>}
            </NavLink>
          ))}
        </nav>

        <button
          onClick={handleLogout}
          className="flex items-center gap-3 px-4 py-3 text-lg hover:bg-white/20 transition rounded-lg m-2 text-white mb-4"
        >
          <FiLogOut size={20} />
          {!isCollapsed && <span>Logout</span>}
        </button>
      </div>
    </div>
  );
}

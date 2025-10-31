// frontend/src/components/Sidebar.jsx
import React, { useEffect } from "react";
import { NavLink } from "react-router-dom";
import { LuBellRing, LuHome, LuCamera, LuUsers, LuFileText } from "react-icons/lu";
import { useUnread } from "../context/UnreadContext";
import { WS_BASE } from "../config";

export default function Sidebar() {
  const { unreadCount, setUnreadCount } = useUnread();

  useEffect(() => {
    const token = localStorage.getItem("token");
    const wsBaseClean = (WS_BASE || window.location.origin.replace(/^http/, "ws")).replace(/\/+$/, "");
    const wsUrl = `${wsBaseClean}/ws${token ? `?token=${encodeURIComponent(token)}` : ""}`;
    let ws;

    try {
      ws = new WebSocket(wsUrl);
    } catch (err) {
      ws = null;
    }

    if (!ws) return;

    ws.onopen = () => {
      // handshake
    };

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        if (!data) return;
        if (Array.isArray(data)) {
          const newViolations = data.filter(d => d && (d.violation_id || d.type === "worker_violation")).length;
          if (newViolations > 0) setUnreadCount((prev) => prev + newViolations);
        } else {
          if (data.violation_id || data.type === "worker_violation") {
            setUnreadCount((prev) => prev + 1);
          }
        }
      } catch (e) {
        // ignore non-json or ack
      }
    };

    ws.onerror = () => {};
    ws.onclose = () => {};

    const handleVisibility = () => {
      if (document.visibilityState === "visible") {
        setUnreadCount((c) => c); // keep unchanged but gives hook a chance to sync UI
      }
    };
    document.addEventListener("visibilitychange", handleVisibility);

    return () => {
      document.removeEventListener("visibilitychange", handleVisibility);
      try {
        ws.close();
      } catch (e) {}
    };
  }, [setUnreadCount]);

  const linkClass = ({ isActive }) =>
    `flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${isActive ? "bg-[#19325C] text-white" : "text-gray-300 hover:bg-[#2A2B30]"}`;

  return (
    <aside className="w-64 bg-[#111215] min-h-screen p-4 border-r border-gray-800">
      <div className="mb-6 px-2">
        <h2 className="text-xl font-bold text-white">Safety Monitor</h2>
        <p className="text-xs text-gray-400">Realtime site monitoring</p>
      </div>

      <nav className="flex flex-col space-y-2">
        <NavLink to="/camera" className={linkClass}>
          <LuCamera />
          <span>Camera</span>
        </NavLink>

        <NavLink to="/notifications" className={linkClass}>
          <div className="relative flex items-center">
            <LuBellRing />
            {unreadCount > 0 && (
              <span className="absolute -top-2 -right-6 inline-flex items-center justify-center px-2 py-0.5 text-xs font-semibold rounded-full bg-red-500 text-white">
                {unreadCount}
              </span>
            )}
          </div>
          <span>Notifications</span>
        </NavLink>

        <NavLink to="/incidents" className={linkClass}>
          <LuHome />
          <span>Incidents</span>
        </NavLink>

        <NavLink to="/workers" className={linkClass}>
          <LuUsers />
          <span>Workers</span>
        </NavLink>

        <NavLink to="/reports" className={linkClass}>
          <LuFileText />
          <span>Reports</span>
        </NavLink>
      </nav>
    </aside>
  );
}

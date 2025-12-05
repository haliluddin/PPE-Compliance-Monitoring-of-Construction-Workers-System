import React, { createContext, useContext, useEffect, useState, useRef } from "react";
import API from "../api";
import { WS_BASE } from "../config";

const UnreadContext = createContext();

export function useUnread() {
  return useContext(UnreadContext);
}

export function UnreadProvider({ children }) {
  const [unreadCount, setUnreadCount] = useState(0);
  const [notifications, setNotifications] = useState([]);
  const originalTitle = useRef(document.title);
  const flashInterval = useRef(null);

  useEffect(() => {
    API.get("/notifications")
      .then((res) => {
        const arr = Array.isArray(res.data) ? res.data : [];
        setNotifications(arr);
        const unread = arr.filter((n) => !n.is_read).length;
        setUnreadCount(unread);
      })
      .catch(() => {});
  }, []);

  function stopFlashingTab() {
    if (flashInterval.current) {
      clearInterval(flashInterval.current);
      flashInterval.current = null;
    }
    document.title = originalTitle.current;
  }

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) return;
    const wsBaseClean = (WS_BASE || window.location.origin.replace(/^http/, "ws")).replace(/\/+$/, "");
    const wsUrl = `${wsBaseClean}/ws/notifications?token=${encodeURIComponent(token)}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {};

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "status_update") {
          setNotifications((prev) =>
            prev.map((n) =>
              (n.violation_id === data.violation_id || n.id === data.violation_id) ? { ...n, status: data.status } : n
            )
          );
          window.dispatchEvent(new CustomEvent("violation_status_update", { detail: data }));
          return;
        }

        if (data.type === "new_violation") {
          setUnreadCount((prev) => prev + 1);
          setNotifications((prev) => [
            {
              id: data.id,
              violation_id: data.violation_id,
              worker_name: data.worker_name,
              worker_code: data.worker_code,
              violation_type: data.violation_type,
              camera: data.camera_display || `${data.camera || "Unknown Camera"} (${data.camera_location || "Unknown Location"})`,
              camera_display: data.camera_display,
              camera_name: data.camera,
              camera_location: data.camera_location,
              is_read: false,
              status: data.status || "Pending",
              created_at: data.created_at || new Date().toISOString(),
              date: data.created_at ? new Date(data.created_at).toLocaleDateString('en-PH') : new Date().toLocaleDateString('en-PH'),
              time: data.created_at ? new Date(data.created_at).toLocaleTimeString('en-PH', { hour: 'numeric', minute: '2-digit', hour12: true }) : new Date().toLocaleTimeString('en-PH', { hour: 'numeric', minute: '2-digit', hour12: true }),
              snapshot: data.snapshot || null,
              type: data.type || "worker_violation"
            },
            ...prev,
          ]);
          window.dispatchEvent(new CustomEvent("violation_new", { detail: data }));
          return;
        }
      } catch (err) {}
    };

    ws.onclose = () => {};
    return () => {
      try { ws.close(); } catch {}
    };
  }, []);

  useEffect(() => {
    if (unreadCount > 0) {
      stopFlashingTab();
      let flash = false;
      if (flashInterval.current) clearInterval(flashInterval.current);
      flashInterval.current = setInterval(() => {
        document.title = flash ? ` (${unreadCount}) Violation Detected!` : originalTitle.current;
        flash = !flash;
      }, 1000);
    } else {
      stopFlashingTab();
    }
    return () => {
      stopFlashingTab();
    };
  }, [unreadCount]);

  const value = {
    unreadCount,
    setUnreadCount,
    notifications,
    setNotifications
  };

  return <UnreadContext.Provider value={value}>{children}</UnreadContext.Provider>;
}

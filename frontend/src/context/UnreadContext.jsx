// frontend/src/context/UnreadContext.jsx
import React, { createContext, useContext, useState, useEffect } from "react";
import API from "../api";

const UnreadContext = createContext();
const audio = new Audio("/notification.mp3");
audio.preload = "auto";

export function UnreadProvider({ children }) {
  const [unreadCount, setUnreadCount] = useState(0);
  const [notifications, setNotifications] = useState([]);

  // 🔊 1️⃣ Unlock audio playback (required in browsers)
  useEffect(() => {
    const unlockAudio = () => {
      audio.play().then(() => {
        audio.pause();
        audio.currentTime = 0;
      }).catch(() => {});
      window.removeEventListener("click", unlockAudio);
    };
    window.addEventListener("click", unlockAudio, { once: true });
  }, []);

  // 🔔 2️⃣ Ask permission for browser notifications
  useEffect(() => {
    if (Notification.permission === "default") {
      Notification.requestPermission();
    }
  }, []);

  // 📨 3️⃣ Load existing notifications from API
  useEffect(() => {
    API.get("/notifications")
      .then((res) => {
        setNotifications(res.data);
        const unread = res.data.filter((n) => !n.is_read).length;
        setUnreadCount(unread);
      })
      .catch(console.error);
  }, []);

  // 🌐 4️⃣ Global WebSocket listener for *real-time* notifications
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) return;

    const ws = new WebSocket(`ws://127.0.0.1:8000/ws/notifications?token=${token}`);

    ws.onopen = () => console.log("🔗 Connected to global notification WebSocket");

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      // 🧠 Determine if this is a *newly created* violation
      // We don’t want to ping for status updates
      const isNewViolation =
        data.violation_id && !notifications.some((n) => n.violation_id === data.violation_id);

      if (isNewViolation) {
        console.log("📢 New violation received:", data);

        // 🔊 Play sound
        audio.currentTime = 0;
        audio.play().catch(console.error);

        // 🖥️ Show browser notification (optional)
        if (Notification.permission === "granted") {
          new Notification("Violation Detected!", {
            body: `${data.worker_name || "Unknown Worker"} - ${
              data.violation_type || data.message
            }`,
            icon: "/alert-icon.png",
          });
        }

        // 🔴 Increment unread count (red dot)
        setUnreadCount((prev) => prev + 1);

        // 🧩 Add to local notifications state
        setNotifications((prev) => [
          {
            id: data.id,
            violation_id: data.violation_id,
            worker_name: data.worker_name,
            worker_code: data.worker_code,
            violation_type: data.violation_type,
            camera: data.camera,
            camera_location: data.camera_location,
            is_read: false,
            status: data.status || "Pending",
            created_at: data.created_at || new Date().toISOString(),
          },
          ...prev,
        ]);
      }
    };

    ws.onclose = () => console.log("🔌 Global WebSocket disconnected");

    // 🧹 Cleanup when component unmounts
    return () => ws.close();

    // ⚠️ Important: don’t include `notifications` as a dependency!
    // Otherwise the socket reopens every time state updates.
  }, []); // 👈 Empty dependency = stable persistent connection

  return (
    <UnreadContext.Provider value={{ unreadCount, setUnreadCount }}>
      {children}
    </UnreadContext.Provider>
  );
}

// Custom hook for components to access unread count
export function useUnread() {
  return useContext(UnreadContext);
}

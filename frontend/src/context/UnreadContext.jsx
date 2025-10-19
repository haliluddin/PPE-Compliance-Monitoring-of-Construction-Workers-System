import React, { createContext, useContext, useState, useEffect, useRef } from "react";
import API from "../api";

const UnreadContext = createContext();

// ✅ Preload notification sound (used only for NEW violations)
const audio = new Audio("/notification.mp3");
audio.preload = "auto";

export function UnreadProvider({ children }) {
  const [unreadCount, setUnreadCount] = useState(0);
  const [notifications, setNotifications] = useState([]);
  const flashInterval = useRef(null);
  const originalTitle = useRef(document.title);

  // ✅ Unlock audio after first user interaction
  useEffect(() => {
    const unlockAudio = () => {
      audio.play()
        .then(() => {
          audio.pause();
          audio.currentTime = 0;
        })
        .catch(() => {});
      window.removeEventListener("click", unlockAudio);
    };
    window.addEventListener("click", unlockAudio, { once: true });
  }, []);

  // ✅ Ask for browser notification permission once
  useEffect(() => {
    if (Notification.permission === "default") {
      Notification.requestPermission();
    }
  }, []);

  // ✅ Load existing notifications
  useEffect(() => {
    API.get("/notifications")
      .then((res) => {
        setNotifications(res.data);
        const unread = res.data.filter((n) => !n.is_read).length;
        setUnreadCount(unread);

        // Stop flashing if nothing unread
        if (unread === 0) stopFlashingTab(true);
      })
      .catch(console.error);
  }, []);

  // ✅ WebSocket setup
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) return;

    const ws = new WebSocket(`ws://127.0.0.1:8000/ws/notifications?token=${token}`);
    ws.onopen = () => console.log("✅ Connected to notification WebSocket");

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      // Silent status update
      if (data.type === "status_update") {
        console.log("ℹ Status update received — silent refresh only.");
        setNotifications((prev) =>
          prev.map((n) =>
            n.violation_id === data.violation_id ? { ...n, status: data.status } : n
          )
        );
        return;
      }

      // New violation
      if (data.type === "new_violation") {
        console.log("🚨 New violation detected:", data);
        audio.currentTime = 0;
        audio.play().catch(console.error);

        if (Notification.permission === "granted") {
          new Notification("Violation Detected!", {
            body: `${data.worker_name || "Unknown Worker"} - ${
              data.violation_type || data.message
            }`,
            icon: "/alert-icon.png",
          });
        }

        setUnreadCount((prev) => prev + 1);
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

        startFlashingTab();
        return;
      }

      console.log("ℹ Ignored unknown event type:", data.type);
    };

    ws.onclose = () => console.log("🔌 Notification WebSocket disconnected");

    return () => {
      ws.close();
      stopFlashingTab(true);
    };
  }, []);

  // ✅ Flash tab title
  const startFlashingTab = () => {
    stopFlashingTab(); // Clear any existing flash
    let flash = false;
    flashInterval.current = setInterval(() => {
      document.title = flash ? "🚨 New Violation Detected!" : originalTitle.current;
      flash = !flash;
    }, 1000);
  };

  // ✅ Stop tab flashing (forceReset = true ensures title resets even if no active interval)
  const stopFlashingTab = (forceReset = false) => {
    if (flashInterval.current) {
      clearInterval(flashInterval.current);
      flashInterval.current = null;
    }
    if (forceReset || document.title !== originalTitle.current) {
      document.title = originalTitle.current;
    }
  };

  // ✅ Stop flashing when tab regains focus
  useEffect(() => {
    const handleFocus = () => stopFlashingTab(true);
    window.addEventListener("focus", handleFocus);
    return () => window.removeEventListener("focus", handleFocus);
  }, []);

  // ✅ NEW: Auto-stop flashing & reset title when all notifs read
  useEffect(() => {
    if (unreadCount === 0) {
      stopFlashingTab(true);
    }
  }, [unreadCount]);

  return (
    <UnreadContext.Provider
      value={{
        unreadCount,
        setUnreadCount,
        notifications,
        setNotifications,
        stopFlashingTab,
      }}
    >
      {children}
    </UnreadContext.Provider>
  );
}

export function useUnread() {
  return useContext(UnreadContext);
}

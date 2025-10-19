
//frontend/src/contect/UnreadContext.jsx
import React, { createContext, useContext, useState, useEffect, useRef } from "react";
import API from "../api";

const UnreadContext = createContext();

// ✅ Preload notification sound (used only for NEW violations)
const audio = new Audio("/notification.mp3");
audio.preload = "auto";

export function UnreadProvider({ children }) {
  const [unreadCount, setUnreadCount] = useState(0);
  const [notifications, setNotifications] = useState([]);
  const flashInterval = useRef(null); // Used for tab flashing
  const originalTitle = useRef(document.title); // Store the original tab title

  // ✅ Unlock audio after first user interaction (browser autoplay rule)
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

  // ✅ Load existing notifications from backend
  useEffect(() => {
    API.get("/notifications")
      .then((res) => {
        setNotifications(res.data);
        const unread = res.data.filter((n) => !n.is_read).length;
        setUnreadCount(unread);
      })
      .catch(console.error);
  }, []);

  // ✅ Global WebSocket setup
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) return;

    const ws = new WebSocket(`ws://127.0.0.1:8000/ws/notifications?token=${token}`);

    ws.onopen = () => console.log("✅ Connected to notification WebSocket");

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      // 🧠 1️⃣ Handle STATUS UPDATE events (silent)
      if (data.type === "status_update") {
        console.log("ℹ Status update received — silent refresh only.");

        // Update the status in current notifications (no sound, flash, or red dot)
        setNotifications((prev) =>
          prev.map((n) =>
            n.violation_id === data.violation_id
              ? { ...n, status: data.status }
              : n
          )
        );

        return; // ❌ Do NOT play sound, flash, or increment unread
      }

      // 🧠 2️⃣ Handle NEW VIOLATION events (trigger alert)
      if (data.type === "new_violation") {
        console.log("🚨 New violation detected:", data);

        // ✅ Play alert sound
        audio.currentTime = 0;
        audio.play().catch(console.error);

        // ✅ Browser popup notification
        if (Notification.permission === "granted") {
          new Notification("Violation Detected!", {
            body: `${data.worker_name || "Unknown Worker"} - ${
              data.violation_type || data.message
            }`,
            icon: "/alert-icon.png",
          });
        }

        // ✅ Increment unread counter
        setUnreadCount((prev) => prev + 1);

        // ✅ Add new notification to local state
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

        // ✅ Flash browser tab to get user's attention
        startFlashingTab();
        return;
      }

      // 🧠 3️⃣ Ignore any other WebSocket message types
      console.log("ℹ Ignored unknown event type:", data.type);
    };

    ws.onclose = () => console.log("🔌 Notification WebSocket disconnected");

    // Cleanup
    return () => {
      ws.close();
      stopFlashingTab();
    };
  }, []);

  // ✅ Flash tab title when a new violation appears
  const startFlashingTab = () => {
    stopFlashingTab(); // Prevent multiple intervals
    let flash = false;

    flashInterval.current = setInterval(() => {
      document.title = flash ? "🚨 New Violation Detected!" : originalTitle.current;
      flash = !flash;
    }, 1000);
  };

  // ✅ Stop tab flashing when user focuses again
  const stopFlashingTab = () => {
    if (flashInterval.current) {
      clearInterval(flashInterval.current);
      flashInterval.current = null;
      document.title = originalTitle.current;
    }
  };

  // ✅ Stop flashing when tab regains focus
  useEffect(() => {
    const handleFocus = () => stopFlashingTab();
    window.addEventListener("focus", handleFocus);
    return () => window.removeEventListener("focus", handleFocus);
  }, []);

  return (
    <UnreadContext.Provider value={{ unreadCount, setUnreadCount, notifications, setNotifications }}>
      {children}
    </UnreadContext.Provider>
  );
}

// ✅ Helper hook for easy usage in any component
export function useUnread() {
  return useContext(UnreadContext);
}

// frontend/src/context/UnreadContext.jsx
import React, { createContext, useContext, useState, useEffect, useRef } from "react";
import API from "../api";

const UnreadContext = createContext();

// Preload the notification sound
const audio = new Audio("/notification.mp3");
audio.preload = "auto";

export function UnreadProvider({ children }) {
  const [unreadCount, setUnreadCount] = useState(0);
  const [notifications, setNotifications] = useState([]);
  const flashInterval = useRef(null); //  Used for flashing the tab
  const originalTitle = useRef(document.title); //  Remember original tab title

  //  Allow browsers to play the sound after first click
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

  //  Ask for browser notification permission once
  useEffect(() => {
    if (Notification.permission === "default") {
      Notification.requestPermission();
    }
  }, []);

  //  Load all existing notifications (for unread count)
  useEffect(() => {
    API.get("/notifications")
      .then((res) => {
        setNotifications(res.data);
        const unread = res.data.filter((n) => !n.is_read).length;
        setUnreadCount(unread);
      })
      .catch(console.error);
  }, []);

  //  Setup a single WebSocket listener (for all pages)
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) return;

    const ws = new WebSocket(`ws://127.0.0.1:8000/ws/notifications?token=${token}`);

    ws.onopen = () => console.log("Connected to global notification WebSocket");

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      //  Ignore updates that aren't "newly created" violations
      const isNewViolation =
        data.violation_id && !notifications.some((n) => n.violation_id === data.violation_id);

      if (!isNewViolation) {
        console.log("ℹ Skipping update (status change or existing notification)");
        return; //  Stop here if it's just a status update
      }

      console.log(" New violation detected:", data);

      //  Play alert sound
      audio.currentTime = 0;
      audio.play().catch(console.error);

      //  Browser notification
      if (Notification.permission === "granted") {
        new Notification("Violation Detected!", {
          body: `${data.worker_name || "Unknown Worker"} - ${
            data.violation_type || data.message
          }`,
          icon: "/alert-icon.png",
        });
      }

      // Increment unread counter
      setUnreadCount((prev) => prev + 1);

      //  Add new notification to local state
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

      //  Start flashing the browser tab
      startFlashingTab();
    };

    ws.onclose = () => console.log(" Global WebSocket disconnected");

    //  Cleanup on unmount
    return () => {
      ws.close();
      stopFlashingTab();
    };
  }, []);

  // Flash the tab title when a new violation is detected
  const startFlashingTab = () => {
    stopFlashingTab(); // Prevent multiple intervals
    let flash = false;

    flashInterval.current = setInterval(() => {
      document.title = flash ? " New Violation Detected!" : originalTitle.current;
      flash = !flash;
    }, 1000);
  };

  //  Stop flashing when user focuses the tab again
  const stopFlashingTab = () => {
    if (flashInterval.current) {
      clearInterval(flashInterval.current);
      flashInterval.current = null;
      document.title = originalTitle.current; // Restore original title
    }
  };

  //  Detect tab focus and stop flashing
  useEffect(() => {
    const handleFocus = () => stopFlashingTab();
    window.addEventListener("focus", handleFocus);
    return () => window.removeEventListener("focus", handleFocus);
  }, []);

  return (
    <UnreadContext.Provider value={{ unreadCount, setUnreadCount }}>
      {children}
    </UnreadContext.Provider>
  );
}

//  Export helper hook for easy use in Header, Sidebar, etc.
export function useUnread() {
  return useContext(UnreadContext);
}

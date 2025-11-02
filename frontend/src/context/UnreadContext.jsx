import React, { createContext, useContext, useState, useEffect, useRef } from "react";
import API from "../api";
import { WS_BASE } from "../config";

const UnreadContext = createContext();

function playNotificationSound() {
  const tryPlay = async () => {
    try {
      const a = new Audio("/notification.mp3");
      a.volume = 0.9;
      await a.play();
    } catch (e) {
      try {
        const Ctx = window.AudioContext || window.webkitAudioContext;
        if (!Ctx) return;
        const ctx = new Ctx();
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.type = "sine";
        o.frequency.value = 880;
        g.gain.value = 0.02;
        o.connect(g);
        g.connect(ctx.destination);
        o.start();
        setTimeout(() => { try { o.stop(); ctx.close(); } catch (err) {} }, 140);
      } catch (err) {}
    }
  };
  tryPlay();
}

export function UnreadProvider({ children }) {
  const [unreadCount, setUnreadCount] = useState(0);
  const [notifications, setNotifications] = useState([]);
  const flashInterval = useRef(null);
  const originalTitle = useRef(document.title);

  useEffect(() => {
    const unlockAudio = () => {
      try {
        const a = new Audio("/notification.mp3");
        a.play().then(() => { a.pause(); a.currentTime = 0; }).catch(() => {});
      } catch (e) {}
      window.removeEventListener("click", unlockAudio);
    };
    window.addEventListener("click", unlockAudio, { once: true });
  }, []);

  useEffect(() => {
    if (Notification.permission === "default") {
      Notification.requestPermission();
    }
  }, []);

  useEffect(() => {
    API.get("/notifications")
      .then((res) => {
        setNotifications(res.data);
        const unread = res.data.filter((n) => !n.is_read).length;
        setUnreadCount(unread);
        if (unread === 0) stopFlashingTab(true);
      })
      .catch(() => {});
  }, []);

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
              n.violation_id === data.violation_id ? { ...n, status: data.status } : n
            )
          );
          return;
        }
        if (data.type === "new_violation") {
          playNotificationSound();
          if (Notification.permission === "granted") {
            new Notification("Violation Detected!", {
              body: `${data.worker_name || "Unknown Worker"} - ${data.violation_type || data.message}`,
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
      } catch (err) {}
    };

    ws.onclose = () => {};
    return () => ws.close();
  }, []);

  const startFlashingTab = () => {
    stopFlashingTab();
    let flash = false;
    flashInterval.current = setInterval(() => {
      document.title = flash ? ` (${unreadCount}) Violation Detected!` : originalTitle.current;
      flash = !flash;
    }, 1000);
  };

  const stopFlashingTab = (forceReset = false) => {
    if (flashInterval.current) {
      clearInterval(flashInterval.current);
      flashInterval.current = null;
    }
    if (forceReset || document.title !== originalTitle.current) {
      document.title = originalTitle.current;
    }
  };

  useEffect(() => {
    const handleFocus = () => stopFlashingTab(true);
    window.addEventListener("focus", handleFocus);
    return () => window.removeEventListener("focus", handleFocus);
  }, []);

  useEffect(() => {
    if (unreadCount > 0) {
      startFlashingTab();
    } else {
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

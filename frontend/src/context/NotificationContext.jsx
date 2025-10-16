// // NotificationContext.jsx
// import { createContext, useContext, useEffect, useState } from "react";
// import API from "../api";

// const NotificationContext = createContext();

// export function NotificationProvider({ children }) {
//   const [notifications, setNotifications] = useState([]);

//   // Compute unread count automatically
//   const unreadCount = notifications.filter((n) => !n.is_read).length;

//   useEffect(() => {
//     API.get("/notifications").then((res) => {
//       // Map to include default is_read
//       const mapped = res.data.map((n) => ({
//         ...n,
//         is_read: n.is_read ?? false
//       }));
//       setNotifications(mapped);
//     });
//   }, []);

//   useEffect(() => {
//     const token = localStorage.getItem("token");
//     if (!token) return;

//     const ws = new WebSocket(`ws://localhost:8000/ws/notifications?token=${token}`);

//     ws.onopen = () => console.log("✅ WebSocket connected for notifications");

//     ws.onmessage = (event) => {
//       const data = JSON.parse(event.data);
//       console.log("🔔 New notification received:", data);

//       // Ensure is_read property exists
//       const newNotification = { ...data, is_read: false };
//       setNotifications((prev) => [newNotification, ...prev]);
//       // no need to manually increment unreadCount
//     };

//     ws.onclose = () => console.log("❌ WebSocket disconnected");

//     return () => ws.close();
//   }, []);

//   return (
//     <NotificationContext.Provider value={{ notifications, setNotifications, unreadCount, setNotifications }}>
//       {children}
//     </NotificationContext.Provider>
//   );
// }

// export function useNotifications() {
//   return useContext(NotificationContext);
// }

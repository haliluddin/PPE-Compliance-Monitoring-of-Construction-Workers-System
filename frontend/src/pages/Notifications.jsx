// frontend/src/pages/Notifications.jsx
import { useEffect, useState } from "react";
import API from "../api";
import { FaExclamationCircle } from "react-icons/fa";
import { FiMoreVertical, FiSearch } from "react-icons/fi";
import { LuBellRing } from "react-icons/lu";
import { useUnread } from "../context/UnreadContext";
import ViolationModal from "../components/ViolationModal";

const audio = new Audio("/notification.mp3");
audio.preload = "auto";


export default function Notifications() {
  const [notifications, setNotifications] = useState([]);
  const [openMenu, setOpenMenu] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const { setUnreadCount } = useUnread();
  const [showModal, setShowModal] = useState(false);
  const [selectedViolation, setSelectedViolation] = useState(null);


  // Filters
  const [cameraOptions, setCameraOptions] = useState([]);
    useEffect(() => {
      const fetchCameras = async () => {
        try {
          const token = localStorage.getItem("token");
          const res = await fetch("http://localhost:8000/cameras/", {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          });
          if (!res.ok) throw new Error("Failed to fetch cameras");
          const data = await res.json();
          setCameraOptions(data.map((c) => c.name)); // Use camera names for the filter
        } catch (err) {
          console.error(err);
        }
      };
      fetchCameras();
    }, []);

  const violationOptions = ["No Helmet", "No Vest", "No Boots", "No Gloves"];
  const sortOptions = ["Newest", "Oldest"];
  const [filters, setFilters] = useState({ camera: "", violation: "", sortBy: "Newest" });

  const handleChange = (filterName, value) => {
    setFilters((prev) => ({ ...prev, [filterName]: value }));
  };

  useEffect(() => {
  const unlockAudio = () => {
    audio.play().then(() => {
      audio.pause(); // immediately pause, now audio is allowed to play
      audio.currentTime = 0;
    }).catch(console.error);
    window.removeEventListener("click", unlockAudio);
  };

  window.addEventListener("click", unlockAudio, { once: true });
}, []);
  // Request browser notification permission
  useEffect(() => {
    const requestPermission = () => {
      if (Notification.permission !== "granted" && Notification.permission !== "denied") {
        Notification.requestPermission().then((perm) => {
          console.log("Notification permission:", perm);
        });
      }
    };
    window.addEventListener("click", requestPermission, { once: true });
    return () => window.removeEventListener("click", requestPermission);
  }, []);

  // Load initial notifications
  useEffect(() => {
    API.get("/notifications").then((res) => {
      const mapped = res.data.map((n) => ({
        id: n.id,
        violation_id: n.violation_id,
        worker: n.worker_name || "Unknown Worker",
        worker_code: n.worker_code || "N/A",
        violation: n.violation_type || n.message || "Unknown Violation",
        camera: `${n.camera || "Unknown Camera"} (${n.camera_location || "Unknown Location"})`,
        camera_name: n.camera || "Unknown Camera",
        type: n.type || "worker_violation",
        date: n.date || new Date(n.created_at).toLocaleDateString(),
        time: n.time || new Date(n.created_at).toLocaleTimeString(),
        isNew: !n.is_read,
        status: n.status || "Pending",
      }));
      setNotifications(mapped);
    });
  }, []);

  // Flash title for unread notifications
  useEffect(() => {
    let originalTitle = document.title;
    let flashInterval;
    const unreadCount = notifications.filter((n) => n.isNew).length;

    if (unreadCount > 0) {
      let showAlert = true;
      flashInterval = setInterval(() => {
        document.title = showAlert ? `(${unreadCount}) Violation Detected!` : originalTitle;
        showAlert = !showAlert;
      }, 1000);
    } else {
      document.title = originalTitle;
    }

    return () => clearInterval(flashInterval);
  }, [notifications]);

  // Update unread count in context
  useEffect(() => {
    const unread = notifications.filter((n) => n.isNew).length;
    setUnreadCount(unread);
  }, [notifications, setUnreadCount]);

  // WebSocket connection for real-time notifications
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) return;

    const ws = new WebSocket(`ws://127.0.0.1:8000/ws/notifications?token=${token}`);

    ws.onopen = () => console.log("Connected to notification WebSocket");

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
const isNewViolation = data.violation_id && !notifications.some(n => n.violation_id === data.violation_id);

      if (isNewViolation) {
        // 🔊 Play sound
        audio.currentTime = 0;
        audio.play().catch(console.error);

        // 🖥️ Browser notification
        if (Notification.permission === "granted") {
          new Notification("Violation Detected!", {
            body: `${data.worker_name || "Unknown Worker"} - ${data.violation_type || data.message}`,
            icon: "/alert-icon.png",
          });
        }
      }
      setNotifications((prev) => {
        if (data.violation_id) {
          const exists = prev.some((n) => n.violation_id === data.violation_id);
          if (exists) {
            // Update existing notification
            return prev.map((n) =>
              n.violation_id === data.violation_id
                ? { ...n, status: data.status || n.status,isNew: isNewViolation ? true : n.isNew }
                : n
            );
          } else {
            // Add new notification
            const newNotification = {
              id: data.id,
              violation_id: data.violation_id,
              worker: data.worker_name || "Unknown Worker",
              worker_code: data.worker_code || "N/A",
              violation: data.violation_type || data.message || "Unknown Violation",
              camera: `${data.camera || "Unknown Camera"} (${data.camera_location || "Unknown Location"})`,
              type:"worker_violation",
              date: new Date(data.created_at).toLocaleDateString(),
              time: new Date(data.created_at).toLocaleTimeString(),
              isNew: true,
              status: data.status || "Pending",
            };
            return [newNotification, ...prev];
          }
        }
        return prev;
      });

      // Update modal if open
      if (selectedViolation && selectedViolation.id === data.violation_id) {
        setSelectedViolation((prev) => ({
          ...prev,
          status: data.status || prev.status,
        }));
      }
    };

    ws.onclose = () => console.log("WebSocket disconnected");
    return () => ws.close();
  }, [audio, selectedViolation]);

  // Toggle notification menu
  const toggleMenu = (id) => setOpenMenu(openMenu === id ? null : id);

  // Mark notification as read
  const markAsRead = async (id) => {
    try {
      await API.post(`/notifications/${id}/mark_read`);
      setNotifications((prev) =>
        prev.map((n) => (n.id === id ? { ...n, isNew: false } : n))
      );
    } catch (err) {
      console.error("Failed to mark notification as read", err);
    }
  };

  const menuActions = [
    { label: "Mark as Read", onClick: markAsRead },
    { label: "Delete Notification", onClick: (id) => alert(`Delete ${id}`) },
    { label: "Report Issue", onClick: (id) => alert(`Report issue for ${id}`) },
  ];

  // Filter and sort notifications
  const filteredNotifications = notifications
    .filter((n) => {
      if (searchQuery) {
      const query = searchQuery.toLowerCase();
      const matchesSearch =
        (n.worker && n.worker.toLowerCase().includes(query)) ||
        (n.violation && n.violation.toLowerCase().includes(query)) ||
        (n.camera && n.camera.toLowerCase().includes(query));
      if (!matchesSearch) return false;
    }
      if (filters.camera && n.camera_name !== filters.camera) return false;
      if (filters.violation && n.violation !== filters.violation) return false;
      if (filter === "unread" && !n.isNew) return false;
      return true;
    })
    .sort((a, b) => {
      if (a.isNew && !b.isNew) return -1;
      if (!a.isNew && b.isNew) return 1;
      if (filters.sortBy === "Oldest") {
        return new Date(a.date + " " + a.time) - new Date(b.date + " " + b.time);
      }
      return new Date(b.date + " " + b.time) - new Date(a.date + " " + a.time);
    });

  return (
    <div className="min-h-screen bg-[#1E1F23] text-gray-100 p-8">
      {/* Page Header */}
      <header className="mb-6 bg-[#2A2B30] px-5 py-3 rounded-xl shadow-lg flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Notifications</h1>
          <p className="text-gray-400 max-w-2xl">
            Below are real-time notifications of safety violations detected on site.
          </p>
        </div>

        {/* Bell Icon with Red Dot */}
        <div className="relative">
          <LuBellRing className="text-[#5388DF]" size={32} />
          {notifications.some((n) => n.isNew) && (
            <span className="absolute top-0 right-0 block h-3 w-3 rounded-full bg-red-500 animate-pulse"></span>
          )}
        </div>
      </header>

      {/* Filters Section */}
      <section className="mb-10">
        <div className="flex flex-col md:flex-row items-stretch md:items-end gap-6">
          <div className="relative w-full md:w-auto flex-1">
            <input
              type="text"
              placeholder="Search Notifications..."
              className="w-full bg-[#2A2B30] text-gray-200 pl-12 pr-4 py-3 rounded-lg border border-gray-700 focus:outline-none focus:border-[#5388DF]"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <FiSearch className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
          </div>

          <div className="flex flex-wrap gap-4">
            <div className="flex flex-col w-48">
              <label className="font-medium text-sm mb-1 text-gray-400">Camera</label>
              <select
                value={filters.camera}
                onChange={(e) => handleChange("camera", e.target.value)}
                className="px-3 py-2 border border-gray-700 rounded-lg bg-[#2A2B30] text-sm text-gray-200 focus:outline-none focus:ring-2 focus:ring-[#5388DF]"
              >
                <option value="">All Cameras</option>
                {cameraOptions.map((cam) => (
                  <option key={cam} value={cam}>
                    {cam}
                  </option>
                ))}
              </select>

            </div>

            <div className="flex flex-col w-48">
              <label className="font-medium text-sm mb-1 text-gray-400">Violation Type</label>
              <select
                value={filters.violation}
                onChange={(e) => handleChange("violation", e.target.value)}
                className="px-3 py-2 border border-gray-700 rounded-lg bg-[#2A2B30] text-sm text-gray-200 focus:outline-none focus:ring-2 focus:ring-[#5388DF]"
              >
                <option value="">All Violations</option>
                {violationOptions.map((vio) => (
                  <option key={vio} value={vio}>
                    {vio}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex flex-col w-48">
              <label className="font-medium text-sm mb-1 text-gray-400">Sort By</label>
              <select
                value={filters.sortBy}
                onChange={(e) => handleChange("sortBy", e.target.value)}
                className="px-3 py-2 border border-gray-700 rounded-lg bg-[#2A2B30] text-sm text-gray-200 focus:outline-none focus:ring-2 focus:ring-[#5388DF]"
              >
                {sortOptions.map((sort) => (
                  <option key={sort} value={sort}>
                    {sort}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </section>

      {/* Notification List */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-3xl font-bold text-white">Notifications</h2>
          <div className="flex space-x-2">
            {[
              { key: "all", label: `All (${notifications.length})` },
              { key: "unread", label: `Unread (${notifications.filter((n) => n.isNew).length})` },
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setFilter(tab.key)}
                className={`px-4 py-1.5 rounded-full text-sm font-medium border transition ${
                  filter === tab.key
                    ? "bg-[#19325C] text-white border-[#19325C]"
                    : "bg-white text-gray-700 border-gray-300 hover:bg-gray-100"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          {filteredNotifications.length === 0 ? (
            <div className="text-center text-gray-500 py-10">No notifications to display.</div>
          ) : (
            filteredNotifications.map((n) => (
              <div key={n.id} className="relative flex bg-[#2A2B30] shadow-lg rounded-xl">
                <div className={`w-2 rounded-l-xl ${n.isNew ? "bg-red-500" : "bg-gray-500"}`}></div>
                <div className="flex flex-1 justify-between items-center p-4 rounded-r-xl">
                  <div className="flex flex-col">
                    <p className="text-white font-semibold text-lg">
                      {n.type === "worker_violation"
                        ? `Worker ${n.worker_code} - ${n.worker}`
                        : n.camera}
                    </p>
                    {n.type === "worker_violation" && (
                      <p className="text-gray-400 text-sm mt-1">{n.camera}</p>
                    )}
                    <div className="flex items-center gap-2 mt-1">
                      <p className="text-gray-300 text-sm">{n.violation}</p>
                      {n.status && (
                        <span
                          className={`inline-flex items-center mt-2 px-2.5 py-1 text-xs font-semibold rounded-full border ${
                            n.status.toLowerCase() === "resolved"
                              ? "bg-green-500/20 text-green-400 border-green-600/50"
                              : n.status.toLowerCase() === "false positive"
                              ? "bg-yellow-500/20 text-yellow-300 border-yellow-600/50"
                              : n.status.toLowerCase() === "pending"
                              ? "bg-red-500/20 text-red-400 border-red-600/50"
                              : "bg-gray-500/20 text-gray-300 border-gray-600/50"
                          }`}
                        >
                          {n.status}
                        </span>
                      )}
                      {n.type === "worker_violation" && <FaExclamationCircle className="text-red-500" size={14} />}
                    </div>
                    <p className="text-gray-500 text-xs mt-1">
                      {n.date} - {n.time}
                    </p>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => {
                        setSelectedViolation({
                          id: n.violation_id,
                          notificationId: n.id,
                          worker: n.worker,
                          worker_code: n.worker_code,
                          violation: n.violation,
                          camera: n.camera,
                          type: n.type,
                          status: n.status,
                          isNew: n.isNew,
                          date: n.date,
                          time: n.time,
                        });
                        setShowModal(true);

                        if (n.isNew) markAsRead(n.id);
                      }}
                      className="px-4 py-2 bg-[#5388DF] text-white rounded-lg text-sm hover:bg-[#19325C] transition"
                    >
                      {n.type === "worker_violation" ? "View Footage" : "View Report"}
                    </button>

                    {/* <div className="relative">
                      <button
                        onClick={() => toggleMenu(n.id)}
                        className="p-2 rounded-full hover:bg-gray-700 transition-colors text-gray-400 hover:text-white"
                      >
                        <FiMoreVertical size={20} />
                      </button>

                      {openMenu === n.id && (
                        <div className="absolute right-0 mt-2 w-48 bg-[#2A2B30] border border-gray-700 rounded-xl shadow-lg z-20">
                          {menuActions.map((action) => (
                            <button
                              key={action.label}
                              onClick={() => {
                                action.onClick(n.id);
                                setOpenMenu(null);
                              }}
                              className="block w-full text-left px-4 py-2 text-sm text-gray-200 hover:bg-gray-700"
                            >
                              {action.label}
                            </button>
                          ))}
                        </div>
                      )}
                    </div> */}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </section>

      {/* Violation Modal */}
      {showModal && selectedViolation && (
        <ViolationModal
          violation={selectedViolation}
          onClose={() => setShowModal(false)}
          onStatusChange={async (newStatus) => {
            try {
              const token = localStorage.getItem("token");

              // Update violation status on backend
              const res = await fetch(
                `http://localhost:8000/violations/${selectedViolation.id}/status`,
                {
                  method: "PUT",
                  headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${token}`,
                  },
                  body: JSON.stringify({ status: newStatus }),
                }
              );
              if (!res.ok) throw new Error("Failed to update status");
              await res.json();

              // Update notifications list locally
              setNotifications((prev) =>
                prev.map((v) =>
                  v.violation_id === selectedViolation.id
                    ? { ...v, status: newStatus, isNew: false }
                    : v
                )
              );

              // Update modal
              setSelectedViolation((prev) => ({ ...prev, status: newStatus }));
            } catch (err) {
              console.error(err);
            }
          }}
        />
      )}
    </div>
  );
}

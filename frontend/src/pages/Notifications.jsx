// frontend/src/pages/Notifications.jsx
import { useEffect, useState } from "react";
import API from "../api";
import { FaExclamationCircle } from "react-icons/fa";
import { FiMoreVertical, FiSearch } from "react-icons/fi";
import { LuBellRing } from "react-icons/lu";
import { useUnread } from "../context/UnreadContext";
import ViolationModal from "../components/ViolationModal";


export default function Notifications() {
  const [notifications, setNotifications] = useState([]);
  const [openMenu, setOpenMenu] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const { setUnreadCount } = useUnread();
const [showModal, setShowModal] = useState(false);
const [selectedViolation, setSelectedViolation] = useState(null);

  // Filters
  const cameraOptions = ["Camera 1", "Camera 2"];
  const violationOptions = ["No Helmet", "No Vest", "No Boots", "No Gloves"];
  const sortOptions = ["Newest", "Oldest"];
  const [filters, setFilters] = useState({
    camera: "",
    violation: "",
    sortBy: "Newest",
  });

  const handleChange = (filterName, value) => {
    setFilters((prev) => ({ ...prev, [filterName]: value }));
  };

  
  const audio = new Audio("/notification.mp3");

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


  useEffect(() => {
    API.get("/notifications").then((res) => {
      const mapped = res.data.map((n) => ({
        id: n.id,
        worker: n.worker_name || "Unknown Worker",
        worker_code: n.worker_code || "N/A",
        violation: n.violation_type || n.message || "Unknown Violation",
        camera: `${n.camera || "Unknown Camera"} (${n.camera_location || "Unknown Location"})`,
        type: n.type || "worker_violation",
        date: n.date || new Date(n.created_at).toLocaleDateString(),
        time: n.time || new Date(n.created_at).toLocaleTimeString(),
        isNew: !n.is_read,
        resolved: n.resolved ?? false,
      }));
      setNotifications(mapped);
    });
  }, []);


  useEffect(() => {
    let originalTitle = document.title;
    let flashInterval;

    const unreadCount = notifications.filter((n) => n.isNew).length;

    if (unreadCount > 0) {
      let showAlert = true;
      flashInterval = setInterval(() => {
        document.title = showAlert
          ? ` (${unreadCount}) Violation Detected!`
          : originalTitle;
        showAlert = !showAlert;
      }, 1000);
    } else {
      document.title = originalTitle;
    }

    return () => clearInterval(flashInterval);
  }, [notifications]);


  useEffect(() => {
    const unread = notifications.filter((n) => n.isNew).length;
    setUnreadCount(unread);
  }, [notifications, setUnreadCount]);

 
  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) return;

    const ws = new WebSocket(`ws://127.0.0.1:8000/ws/notifications?token=${token}`);

    ws.onopen = () => console.log("Connected to notification WebSocket");

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("New notification received:", data);

      
      audio.play().catch((err) => console.error("Audio play failed:", err));

      
      if (document.hidden && Notification.permission === "granted") {
        new Notification("Violation Detected!", {
          body: `${data.worker_name || "Unknown Worker"} - ${data.violation_type || data.message}`,
          icon: "/alert-icon.png",
        });
      }

      const newNotification = {
        id: data.id,
        worker: data.worker_name || "Unknown Worker",
        worker_code: data.worker_code || "N/A",
        violation: data.violation_type || data.message || "Unknown Violation",
        camera: `${data.camera || "Unknown Camera"} (${data.camera_location || "Unknown Location"})`,
        type: data.type || "worker_violation",
        date: new Date(data.created_at).toLocaleDateString(),
        time: new Date(data.created_at).toLocaleTimeString(),
        isNew: true,
        resolved: data.resolved ?? false,
      };

      setNotifications((prev) => [newNotification, ...prev]);
    };

    ws.onclose = () => console.log("WebSocket disconnected");

    return () => ws.close();
  }, [audio]);

  const toggleMenu = (id) => setOpenMenu(openMenu === id ? null : id);

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

  const filteredNotifications = notifications
    .filter((n) => {
      if (
        searchQuery &&
        !(
          (n.worker && n.worker.toLowerCase().includes(searchQuery.toLowerCase())) ||
          (n.violation && n.violation.toLowerCase().includes(searchQuery.toLowerCase())) ||
          (n.camera && n.camera.toLowerCase().includes(searchQuery.toLowerCase()))
        )
      )
        return false;
      if (filters.camera && n.camera !== filters.camera) return false;
      if (filters.violation && n.violation !== filters.violation) return false;
      if (filter === "unread" && !n.isNew) return false;
      return true;
    })
    .sort((a, b) => {
      // Unread at top
      if (a.isNew && !b.isNew) return -1;
      if (!a.isNew && b.isNew) return 1;

      // Then sort by chosen filter
      if (filters.sortBy === "Oldest") {
        return new Date(a.date + " " + a.time) - new Date(b.date + " " + b.time);
      } else {
        return new Date(b.date + " " + b.time) - new Date(a.date + " " + a.time);
      }
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
        <LuBellRing className="text-[#5388DF]" size={32} />
      </header>

      {/* Filters */}
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
                  <option key={cam} value={cam}>{cam}</option>
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
                  <option key={vio} value={vio}>{vio}</option>
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
                  <option key={sort} value={sort}>{sort}</option>
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
                className={`px-4 py-1.5 rounded-full text-sm font-medium border transition
                ${filter === tab.key
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
                <div className={`w-2 rounded-l-xl ${n.isNew ? 'bg-red-500' : 'bg-gray-500'}`}></div>

                <div className="flex flex-1 justify-between items-center p-4 rounded-r-xl">
                  <div className="flex flex-col">
                    <p className="text-white font-semibold text-lg">
                      {n.type === "worker_violation" ? `Worker ${n.worker_code} - ${n.worker}` : n.camera}
                    </p>
                    {n.type === "worker_violation" && <p className="text-gray-400 text-sm mt-1">{n.camera}</p>}
                    <div className="flex items-center gap-2 mt-1">
                      <p className="text-gray-300 text-sm">{n.violation}</p>
                      {n.type === "worker_violation" && <FaExclamationCircle className="text-red-500" size={14} />}
                    </div>
                    <p className="text-gray-500 text-xs mt-1">{n.date} - {n.time}</p>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => {
                        setSelectedViolation(n);
                        setShowModal(true);
                      }}
                      className="px-4 py-2 bg-[#5388DF] text-white rounded-lg text-sm hover:bg-[#19325C] transition"
                    >
                      {n.type === "worker_violation" ? "View Footage" : "View Report"}
                    </button>

                    <div className="relative">
                      <button onClick={() => toggleMenu(n.id)} className="p-2 rounded-full hover:bg-gray-700 transition-colors text-gray-400 hover:text-white">
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
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </section>

      {showModal && selectedViolation && (
        <ViolationModal
          violation={selectedViolation}
          onClose={() => setShowModal(false)}
          onStatusChange={async (newStatus) => {
            try {
              const token = localStorage.getItem("token");

            
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
              const data = await res.json();

          
              await API.post(`/notifications/${selectedViolation.id}/mark_read`);

              
              setNotifications((prev) =>
                prev.map((v) =>
                  v.id === selectedViolation.id
                    ? { ...v, status: data.status, isNew: false }
                    : v
                )
              );

              const channel = new BroadcastChannel("violations_update");
              channel.postMessage({
                id: selectedViolation.id,
                status: data.status,
              });

              
              setSelectedViolation((prev) => ({
                ...prev,
                status: data.status,
              }));
            } catch (err) {
              console.error("Error updating status or marking as read:", err);
            }
          }}
        />
      )}


    </div>
  );
}

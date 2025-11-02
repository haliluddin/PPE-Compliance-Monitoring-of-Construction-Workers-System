import { useEffect, useState } from "react";
import API from "../api";
import { FaExclamationCircle } from "react-icons/fa";
import { FiMoreVertical, FiSearch } from "react-icons/fi";
import { LuBellRing } from "react-icons/lu";
import { useUnread } from "../context/UnreadContext";
import ViolationModal from "../components/ViolationModal";
import { WS_BASE } from "../config";

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

export default function Notifications() {
  const [notifications, setNotifications] = useState([]);
  const [openMenu, setOpenMenu] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const { setUnreadCount } = useUnread();
  const [showModal, setShowModal] = useState(false);
  const [selectedViolation, setSelectedViolation] = useState(null);
  const [cameraOptions, setCameraOptions] = useState([]);
  const violationOptions = ["No Helmet", "No Vest", "No Boots", "No Gloves"];
  const sortOptions = ["Newest", "Oldest"];
  const [filters, setFilters] = useState({ camera: "", violation: "", sortBy: "Newest" });

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
    const requestPermission = () => {
      if (Notification.permission !== "granted" && Notification.permission !== "denied") {
        Notification.requestPermission().then(() => {});
      }
    };
    window.addEventListener("click", requestPermission, { once: true });
    return () => window.removeEventListener("click", requestPermission);
  }, []);

  useEffect(() => {
    const fetchCameras = async () => {
      try {
        const res = await API.get("/cameras");
        const data = res.data;
        setCameraOptions((Array.isArray(data) ? data : data?.cameras || []).map((c) => c.name || c.location || `Camera ${c.id}`));
      } catch (err) {}
    };
    fetchCameras();
  }, []);

  useEffect(() => {
    const loadNotifications = async () => {
      try {
        const res = await API.get("/notifications");
        const raw = res.data;
        const arr = Array.isArray(raw) ? raw : raw?.notifications || raw;
        const mapped = (arr || []).map((n) => ({
          id: n.id,
          violation_id: n.violation_id || n.id,
          worker: n.worker_name || n.worker || "Unknown Worker",
          worker_code: n.worker_code || "N/A",
          violation: n.violation_type || n.violation || n.message || "Unknown Violation",
          camera: `${n.camera || n.camera_name || "Unknown Camera"} (${n.camera_location || ""})`,
          camera_name: n.camera || n.camera_name || "Unknown Camera",
          type: n.type || "worker_violation",
          date: n.date || (n.created_at ? new Date(n.created_at).toLocaleDateString() : ""),
          time: n.time || (n.created_at ? new Date(n.created_at).toLocaleTimeString() : ""),
          isNew: typeof n.is_read !== "undefined" ? !n.is_read : !(n.is_read === true),
          status: n.status || "Pending",
        }));
        setNotifications(mapped);
      } catch (err) {}
    };
    loadNotifications();
  }, []);

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

  useEffect(() => {
    const unread = notifications.filter((n) => n.isNew).length;
    setUnreadCount(unread);
  }, [notifications, setUnreadCount]);

  useEffect(() => {
    const token = localStorage.getItem("token");
    if (!token) return;
    const wsBaseClean = (WS_BASE || window.location.origin.replace(/^http/, "ws")).replace(/\/+$/, "");
    const wsUrl = `${wsBaseClean}/ws${token ? `?token=${encodeURIComponent(token)}` : ""}`;
    const ws = new WebSocket(wsUrl);
    ws.onopen = () => {};
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const isNewViolation = data.violation_id && !notifications.some((n) => n.violation_id === data.violation_id);
        if (isNewViolation) {
          playNotificationSound();
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
              return prev.map((n) =>
                n.violation_id === data.violation_id
                  ? { ...n, status: data.status || n.status, isNew: isNewViolation ? true : n.isNew }
                  : n
              );
            } else {
              const newNotification = {
                id: data.id,
                violation_id: data.violation_id,
                worker: data.worker_name || "Unknown Worker",
                worker_code: data.worker_code || "N/A",
                violation: data.violation_type || data.message || "Unknown Violation",
                camera: `${data.camera || "Unknown Camera"} (${data.camera_location || "Unknown Location"})`,
                camera_name: data.camera || "Unknown Camera",
                type: "worker_violation",
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
      } catch (err) {}
    };
    ws.onclose = () => {};
    return () => ws.close();
  }, [notifications]);

  const toggleMenu = (id) => setOpenMenu(openMenu === id ? null : id);

  const markAsRead = async (id) => {
    try {
      await API.post(`/notifications/${id}/mark_read`);
      setNotifications((prev) => prev.map((n) => (n.id === id ? { ...n, isNew: false } : n)));
    } catch (err) {}
  };

  const normalizeViolationForModal = (n) => {
    const snapshot = n.snapshot && typeof n.snapshot === "object"
      ? (n.snapshot.base64 || n.snapshot.data || n.snapshot)
      : n.snapshot;
    const created_at = n.created_at || n.date || new Date().toISOString();
    const workerField = n.worker || n.worker_name || n.worker_code || "Unknown Worker";
    let workerStr = "Unknown Worker";
    if (typeof workerField === "string") workerStr = workerField;
    else if (typeof workerField === "object") {
      workerStr = workerField.fullName || workerField.name || workerField.firstName || `${workerField.first || ""} ${workerField.last || ""}`.trim() || JSON.stringify(workerField);
    } else workerStr = String(workerField);
    return {
      id: n.violation_id || n.id,
      notificationId: n.id,
      worker: workerStr,
      worker_code: n.worker_code,
      violation: n.violation,
      camera: n.camera,
      type: n.type,
      status: n.status,
      isNew: n.isNew,
      created_at,
      snapshot,
    };
  };

  const handleChange = (filterName, value) => {
    setFilters((prev) => ({ ...prev, [filterName]: value }));
  };

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
      <header className="mb-6 bg-[#2A2B30] px-5 py-3 rounded-xl shadow-lg flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Notifications</h1>
          <p className="text-gray-400 max-w-2xl">Below are real-time notifications of safety violations detected on site.</p>
        </div>
        <div className="relative">
          <LuBellRing className="text-[#5388DF]" size={32} />
          {notifications.some((n) => n.isNew) && (
            <span className="absolute top-0 right-0 block h-3 w-3 rounded-full bg-red-500 animate-pulse"></span>
          )}
        </div>
      </header>

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
                className={`px-4 py-1.5 rounded-full text-sm font-medium border transition ${filter === tab.key ? "bg-[#19325C] text-white border-[#19325C]" : "bg-white text-gray-700 border-gray-300 hover:bg-gray-100"}`}
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
                      {n.type === "worker_violation" ? `Worker ${n.worker_code} - ${n.worker}` : n.camera}
                    </p>
                    {n.type === "worker_violation" && <p className="text-gray-400 text-sm mt-1">{n.camera}</p>}
                    <div className="flex items-center gap-2 mt-1">
                      <p className="text-gray-300 text-sm">{n.violation}</p>
                      {n.status && (
                        <span className={`inline-flex items-center mt-2 px-2.5 py-1 text-xs font-semibold rounded-full border ${n.status.toLowerCase() === "resolved" ? "bg-green-500/20 text-green-400 border-green-600/50" : n.status.toLowerCase() === "false positive" ? "bg-yellow-500/20 text-yellow-300 border-yellow-600/50" : "bg-red-500/20 text-red-400 border-red-600/50"}`}>
                          {n.status}
                        </span>
                      )}
                      {n.type === "worker_violation" && <FaExclamationCircle className="text-red-500" size={14} />}
                    </div>
                    <p className="text-gray-500 text-xs mt-1">{n.date} - {n.time}</p>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => {
                        const norm = normalizeViolationForModal(n);
                        setSelectedViolation(norm);
                        setShowModal(true);
                        if (n.isNew) markAsRead(n.id);
                      }}
                      className="px-4 py-2 bg-[#5388DF] text-white rounded-lg text-sm hover:bg-[#19325C] transition"
                    >
                      {n.type === "worker_violation" ? "View Footage" : "View Report"}
                    </button>
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
              const res = await fetch(`${API_BASE || ""}/violations/${selectedViolation.id}/status`, {
                method: "PUT",
                headers: {
                  "Content-Type": "application/json",
                  Authorization: `Bearer ${token}`,
                },
                body: JSON.stringify({ status: newStatus }),
              });
              if (!res.ok) throw new Error("Failed to update status");
              const data = await res.json();
              setNotifications((prev) =>
                prev.map((v) =>
                  v.violation_id === selectedViolation.id ? { ...v, status: data.status || newStatus, isNew: false } : v
                )
              );
              setSelectedViolation((prev) => ({ ...prev, status: newStatus }));
            } catch (err) {}
          }}
        />
      )}
    </div>
  );
}

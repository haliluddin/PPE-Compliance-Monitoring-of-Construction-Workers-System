import { useState } from "react";
import { FaExclamationCircle } from "react-icons/fa";
import { FiMoreVertical } from "react-icons/fi";

export default function Incident() {
  const notifications = [
    {
      id: 1,
      camera: "Camera A",
      violation: "No Helmet",
      worker: "Worker 1",
      isNew: true,
      date: "2025-09-27",
    },
    {
      id: 2,
      camera: "Camera B",
      violation: "No Vest",
      worker: "Worker 2",
      isNew: false,
      date: "2025-09-26",
    },
  ];

  const [openMenu, setOpenMenu] = useState(null);
  const [filter, setFilter] = useState("all");

  const toggleMenu = (id) => {
    setOpenMenu(openMenu === id ? null : id);
  };

  const menuActions = [
    { label: "Mark as Unread", onClick: (id) => alert(`Mark ${id} as unread`) },
    { label: "Delete Notification", onClick: (id) => alert(`Delete ${id}`) },
    { label: "Report Issue", onClick: (id) => alert(`Report issue for ${id}`) },
  ];

  const filteredNotifications =
    filter === "unread"
      ? notifications.filter((n) => n.isNew)
      : notifications;

  return (
    <div className="min-h-screen bg-gray-50 text-[#19325C] p-4">
      {/* ---------- Page Header ---------- */}
      <header className="mb-6">
        <p className="text-gray-600 max-w-2xl">
          Below are real-time notifications of safety violations detected on site.
        </p>
      </header>

      {/* ---------- Filters ---------- */}
      <section className="mb-8">
        <div className="flex flex-wrap gap-6">
          {["Camera", "Violation Type"].map((label) => (
            <div className="flex flex-col w-60" key={label}>
              <label className="font-medium text-sm mb-1">{label}</label>
              <select className="px-3 py-2 border border-gray-300 rounded-lg bg-white shadow-sm text-sm focus:outline-none focus:ring-2 focus:ring-[#19325C]">
                <option>Select an option</option>
                <option>Placeholder</option>
                <option>Placeholder</option>
              </select>
              <span className="text-xs text-gray-500 mt-1">
                Pick a {label.toLowerCase()} to filter.
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* ---------- Notification list ---------- */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Notifications</h2>

          {/* All/unread toggle */}
          <div className="flex space-x-2">
            {[
              { key: "all", label: `All (${notifications.length})` },
              {
                key: "unread",
                label: `Unread (${notifications.filter((n) => n.isNew).length})`,
              },
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setFilter(tab.key)}
                className={`px-4 py-1.5 rounded-full text-sm font-medium border transition 
                ${
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

        <div className="space-y-2">
          {filteredNotifications.length === 0 ? (
            <div className="text-center text-gray-500 py-10">
              No notifications to display.
            </div>
          ) : (
            filteredNotifications.map((n) => (
              <div
                key={n.id}
                className="relative flex justify-between items-start p-4 bg-white rounded-xl border border-gray-200 hover:bg-gray-50 transition"
              >
                {/* Left: unread dot + text */}
                <div className="flex gap-3">
                  {/* Blue dot for unread */}
                  {n.isNew && (
                    <span className="w-3 h-3 mt-2 rounded-full bg-blue-500 shrink-0"></span>
                  )}

                  <div className="flex flex-col">
                    <p className="text-sm text-gray-800 leading-snug">
                      <span className="font-semibold">{n.worker}</span> was
                      detected by <span className="font-medium">{n.camera}</span>{" "}
                      for a <span className="font-semibold">{n.violation}</span>{" "}
                      violation.
                    </p>

                    <div className="flex items-center gap-2 mt-1 text-xs text-gray-500">
                      <FaExclamationCircle className="text-red-500" size={12} />
                      <span>{n.date}</span>
                    </div>
                  </div>
                </div>

                {/* Right: menu */}
                <div className="relative">
                  <button
                    onClick={() => toggleMenu(n.id)}
                    className="p-2 rounded-full hover:bg-gray-100 transition-colors"
                    aria-label="Actions menu"
                  >
                    <FiMoreVertical size={20} />
                  </button>

                  {openMenu === n.id && (
                    <div className="absolute right-0 mt-2 w-48 bg-white border border-gray-200 rounded-xl shadow-lg z-20">
                      {menuActions.map((action) => (
                        <button
                          key={action.label}
                          onClick={() => {
                            action.onClick(n.id);
                            setOpenMenu(null);
                          }}
                          className="block w-full text-left px-4 py-2 text-sm hover:bg-gray-100"
                        >
                          {action.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </section>
    </div>
  );
}

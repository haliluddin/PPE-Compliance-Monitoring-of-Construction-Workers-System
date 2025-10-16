import { useEffect, useState } from "react";
import API from "../api";
import { FaExclamationCircle } from "react-icons/fa";
import { FiMoreVertical, FiSearch, FiFilter } from "react-icons/fi";
import { LuBellRing } from "react-icons/lu";

export default function Notifications() {
  const [notifications, setNotifications] = useState([]);
  const [openMenu, setOpenMenu] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
 const [filter, setFilter] = useState("all");


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


  
  const toggleMenu = (id) => {
    setOpenMenu(openMenu === id ? null : id);
  };

  const handleChange = (filterName, value) => {
    setFilters((prevFilters) => ({
      ...prevFilters,
      [filterName]: value,
    }));
  };

   /* dropdown lists */
    const cameraOptions   = ["Camera 1", "Camera 2"]; // Updated to reflect existing cameras
    const violationOptions = ["No Helmet", "No Vest", "No Boots", "No Gloves"];
    
    const sortOptions = ["Newest", "Oldest"];
  
    /* selected value */
    const [filters, setFilters] = useState({
      camera: "",
      violation: "",
      sortBy: "Newest",
    });
    
  const menuActions = [
    { label: "Mark as Unread", onClick: (id) => alert(`Mark ${id} as unread`) },
    { label: "Delete Notification", onClick: (id) => alert(`Delete ${id}`) },
    { label: "Report Issue", onClick: (id) => alert(`Report issue for ${id}`) },
  ];

  const filteredNotifications = notifications
    .filter((n) => {
      // Filter by search query
      if (searchQuery &&
          !(n.worker && n.worker.toLowerCase().includes(searchQuery.toLowerCase())) &&
          !(n.violation && n.violation.toLowerCase().includes(searchQuery.toLowerCase())) &&
          !(n.camera && n.camera.toLowerCase().includes(searchQuery.toLowerCase()))
      ) {
        return false;
      }

      // Filter by camera
      if (filters.camera && n.camera !== filters.camera) {
        return false;
      }

      // Filter by violation
      if (filters.violation && n.violation !== filters.violation) {
        return false;
      }

      // Filter by unread/all toggle
      if (filter === 'unread' && !n.isNew) {
        return false;
      }

      return true;
    })
    .sort((a, b) => {
      if (filters.sortBy === 'Oldest') {
        return new Date(a.date + ' ' + a.time) - new Date(b.date + ' ' + b.time);
      } else {
        return new Date(b.date + ' ' + b.time) - new Date(a.date + ' ' + a.time);
      }
    });

  return (
    <div className="min-h-screen bg-[#1E1F23] text-gray-100 p-8">
      
      

      {/* ---------- Page Header ---------- */}
      <header className="mb-6 bg-[#2A2B30] px-5 py-3 rounded-xl shadow-lg flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Notifications</h1>
          <p className="text-gray-400 max-w-2xl">
            Below are real-time notifications of safety violations detected on site.
          </p>
        </div>
        <LuBellRing className="text-[#5388DF]" size={32} />
      </header>

      {/* ---------- Filters ---------- */}
      <section className="mb-10">
        <div className="flex flex-col md:flex-row items-stretch md:items-end gap-6">
          {/* Search Bar */}
          <div className="relative w-full md:w-auto flex-1">
            <label htmlFor="search-notifications" className="font-medium text-sm mb-1 text-gray-400 sr-only">Search Notifications</label>
            <input
              type="text"
              id="search-notifications"
              placeholder="Search Notifications..."
              className="w-full bg-[#2A2B30] text-gray-200 pl-12 pr-4 py-3 rounded-lg border border-gray-700 focus:outline-none focus:border-[#5388DF]"
              onChange={(e) => setSearchQuery(e.target.value)}
              value={searchQuery}
            />
            <FiSearch 
              className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" 
              size={20} 
            />
          </div>

          {/* Filter Dropdowns */}
          <div className="flex flex-wrap gap-4">
            {/* Camera Dropdown */}
            <div className="flex flex-col w-48">
              <label className="font-medium text-sm mb-1 text-gray-400">Camera</label>
              <select
                value={filters.camera}
                onChange={(e) => handleChange("camera", e.target.value)}
                className="px-3 py-2 border border-gray-700 rounded-lg bg-[#2A2B30] shadow-sm text-sm text-gray-200 focus:outline-none focus:ring-2 focus:ring-[#5388DF]"
              >
                <option value="">All Cameras</option>
                {cameraOptions.map((cam) => (
                  <option key={cam} value={cam}>{cam}</option>
                ))}
              </select>
            </div>

            {/* Violation Type Dropdown */}
            <div className="flex flex-col w-48">
              <label className="font-medium text-sm mb-1 text-gray-400">Violation Type</label>
              <select
                value={filters.violation}
                onChange={(e) => handleChange("violation", e.target.value)}
                className="px-3 py-2 border border-gray-700 rounded-lg bg-[#2A2B30] shadow-sm text-sm text-gray-200 focus:outline-none focus:ring-2 focus:ring-[#5388DF]"
              >
                <option value="">All Violations</option>
                {violationOptions.map((vio) => (
                  <option key={vio} value={vio}>{vio}</option>
                ))}
              </select>
            </div>

            {/* Sort By Dropdown */}
            <div className="flex flex-col w-48">
              <label className="font-medium text-sm mb-1 text-gray-400">Sort By</label>
              <select
                value={filters.sortBy}
                onChange={(e) => handleChange("sortBy", e.target.value)}
                className="px-3 py-2 border border-gray-700 rounded-lg bg-[#2A2B30] shadow-sm text-sm text-gray-200 focus:outline-none focus:ring-2 focus:ring-[#5388DF]"
              >
                {sortOptions.map((sort) => (
                  <option key={sort} value={sort}>{sort}</option>
                ))}
              </select>
            </div>

          </div>

        </div>
      </section>

      {/* ---------- Notification list ---------- */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-3xl font-bold text-white">Notifications</h2>

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

        <div className="space-y-4">
          {filteredNotifications.length === 0 ? (
            <div className="text-center text-gray-500 py-10">
              No notifications to display.
            </div>
          ) : (
            filteredNotifications.map((n) => (
              <div
                key={n.id}
                className="relative flex bg-[#2A2B30] shadow-lg rounded-xl"
              >
                {/* Colored Left Bar */}
                <div
                  className={`w-2 rounded-l-xl ${n.isNew ? 'bg-red-500' : 'bg-gray-500'}`}
                ></div>

                <div className="flex flex-1 justify-between items-center p-4 rounded-r-xl">
                  {/* Left Section: Image and Text */}
                  <div className="flex items-center gap-4">
                    {/* Worker/Camera Image */}
                    

                    <div className="flex flex-col">
                      {/* Title */}
                      {n.type === 'worker_violation' ? (
                       <p className="text-white font-semibold text-lg">
                       Worker {n.worker_code} - {n.worker}
                      </p>

                      ) : (
                        <p className="text-white font-semibold text-lg">
                          {n.camera}
                        </p>
                      )}
                      {n.type === 'worker_violation' && (
                        <p className="text-gray-400 text-sm mt-1">
                          {n.camera}
                        </p>
                      )}

                      {/* Violation/Alert Description */}
                      <div className="flex items-center gap-2 mt-1">
                        <p className="text-gray-300 text-sm">
                          {n.violation}
                        </p>
                        {n.type === 'worker_violation' && (
                          <FaExclamationCircle className="text-red-500" size={14} />
                        )}
                        {n.type === 'camera_alert' && n.resolved && (
                          <span className="text-green-500"><svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 inline-block ml-1" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 13.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/></svg></span>
                        )}
                      </div>

                      {/* Date and Time */}
                      <p className="text-gray-500 text-xs mt-1">
                        {n.date} - {n.time}
                      </p>

                      {/* Resolved by System */}
                      {n.type === 'camera_alert' && n.resolved && (
                        <p className="text-green-500 text-xs mt-1 flex items-center gap-1">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 13.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"/></svg> Resolved by System (Auto-clear)
                        </p>
                      )}
                    </div>
                  </div>

                  {/* Right Section: Action Buttons and Menu */}
                  <div className="flex items-center gap-2">
                    {n.type === 'worker_violation' && (
                      <>
                        <button className="px-4 py-2 bg-[#5388DF] text-white rounded-lg text-sm hover:bg-[#19325C] transition">
                          View Footage
                        </button>
                      </>
                    )}
                    {n.type === 'camera_alert' && (
                      <button className="px-4 py-2 bg-[#5388DF] text-white rounded-lg text-sm hover:bg-[#19325C] transition">
                        View Report
                      </button>
                    )}
                    
                    <div className="relative">
                      <button
                        onClick={() => toggleMenu(n.id)}
                        className="p-2 rounded-full hover:bg-gray-700 transition-colors text-gray-400 hover:text-white"
                        aria-label="Actions menu"
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
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </section>
    </div>
  );
}
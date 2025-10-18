//frontend/src/pages/Incident.jsx
import React, { useState, useEffect } from "react";
import { FaEye } from "react-icons/fa";
import { FiSearch, FiCalendar } from "react-icons/fi";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import { format } from "date-fns";
import { API_BASE } from "../config";   // <-- add/remove this line as needed
import ViolationModal from "../components/ViolationModal";

export default function Incident() {
  const [notifications, setNotifications] = useState([]);
  const [filters, setFilters] = useState({
    camera: "",
    violation: "",
    date: null,
    sortBy: "newest",
  });
  const [searchQuery, setSearchQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(5);

  const [showModal, setShowModal] = useState(false);
const [selectedViolation, setSelectedViolation] = useState(null);
const [updating, setUpdating] = useState(false);


  useEffect(() => {
    const token = localStorage.getItem("token");
    fetch("http://localhost:8000/violations/", {
    //fetch(`${API_BASE}/violations/`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
      .then((res) => {
        if (!res.ok) throw new Error("Unauthorized");
        return res.json();
      })
      .then((data) => setNotifications(data))
      .catch((err) => console.error("Error fetching violations:", err));
  }, []);

  const handleChange = (key, value) =>
    setFilters((prev) => ({ ...prev, [key]: value }));
  const handleDateChange = (date) => handleChange("date", date);

 const [cameraOptions, setCameraOptions] = useState([]);

useEffect(() => {
  const token = localStorage.getItem("token");
  fetch("http://localhost:8000/cameras/", {
  //fetch(`${API_BASE}/cameras/`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  })
    .then((res) => {
      if (!res.ok) throw new Error("Unauthorized");
      return res.json();
    })
    .then((data) => setCameraOptions(data))
    .catch((err) => console.error("Error fetching cameras:", err));
}, []);

  const violationOptions = [
    "No Helmet",
    "No Vest",
    "No Gloves",
    "No Safety Shoes",
  ];

  const filteredNotifications = notifications
    .filter((n) => {
      if (
        searchQuery &&
        !(
          (n.worker &&
            n.worker.toLowerCase().includes(searchQuery.toLowerCase())) ||
          (n.violation &&
            n.violation.toLowerCase().includes(searchQuery.toLowerCase())) ||
          (n.camera &&
            n.camera.toLowerCase().includes(searchQuery.toLowerCase()))
        )
      ) {
        return false;
      }
      if (filters.camera && n.camera !== filters.camera) return false;
      if (filters.violation && n.violation !== filters.violation) return false;
      if (filters.date) {
        const notificationDate = new Date(n.date.split(" ")[0]);
        const filterDate = new Date(filters.date);
        if (notificationDate.toDateString() !== filterDate.toDateString())
          return false;
      }
      return true;
    })
    .sort((a, b) => {
      const dateA = new Date(a.date);
      const dateB = new Date(b.date);
      return filters.sortBy === "newest" ? dateB - dateA : dateA - dateB;
    });

  const totalPages = Math.ceil(filteredNotifications.length / itemsPerPage);
  const currentItems = filteredNotifications.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case "resolved":
        return "bg-green-500/20 text-green-400 border-green-600/50";
      case "false positive":
        return "bg-yellow-500/20 text-yellow-300 border-yellow-600/50";
      case "pending":
      default:
        return "bg-red-500/20 text-red-400 border-red-600/50";
    }
  };

  // Handle open modal
  const handleView = (violation) => {
    setSelectedViolation(violation);
    setShowModal(true);
  };

  const handleStatusChange = async (newStatus) => {
  if (!selectedViolation) return;
  setUpdating(true);
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

    setNotifications((prev) =>
      prev.map((v) =>
        v.id === selectedViolation.id ? { ...v, status: data.status } : v
      )
    );
    setSelectedViolation((prev) => ({ ...prev, status: data.status }));
  } catch (err) {
    console.error("Error updating violation status:", err);
  } finally {
    setUpdating(false);
  }
};


  return (
    <div className="min-h-screen bg-[#1E1F23] text-gray-100 p-6">
      <div className="bg-[#2A2B30] px-5 py-3 rounded-xl shadow-lg mb-8 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 text-gray-300">
            <span className="text-lg font-semibold">Total Incidents:</span>
            <span className="text-xl font-bold text-[#5388DF]">
              {notifications.length}
            </span>
          </div>
        </div>
      </div>

      {/* Filters */}
      <section className="mb-10">
        <div className="flex flex-col md:flex-row items-stretch md:items-center gap-6">
          {/* Search */}
          <div className="relative w-full md:w-auto flex-1">
            <input
              type="text"
              placeholder="Search Incidents..."
              className="w-full bg-[#2A2B30] text-gray-200 pl-12 pr-4 py-3 rounded-lg border border-gray-700 focus:outline-none focus:border-[#5388DF]"
              onChange={(e) => setSearchQuery(e.target.value)}
              value={searchQuery}
            />
            <FiSearch
              className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400"
              size={20}
            />
          </div>

          {/* Camera */}
          <div className="flex flex-col w-40">
            <label className="font-medium text-sm mb-1 text-gray-400">
              Camera Location
            </label>
            <select
              value={filters.camera}
              onChange={(e) => handleChange("camera", e.target.value)}
              className="px-3 py-3 border border-gray-700 rounded-lg bg-[#2A2B30] shadow-sm text-sm focus:outline-none focus:ring-2 focus:ring-[#5388DF] text-gray-200"
            >
              <option value="">All Camera</option>
              {cameraOptions.map((cam) => (
                <option key={cam.id} value={cam.name}>
                  {cam.name} — {cam.location}
                </option>
              ))}

            </select>
          </div>

          {/* Violation */}
          <div className="flex flex-col w-40">
            <label className="font-medium text-sm mb-1 text-gray-400">
              Violation Type
            </label>
            <select
              value={filters.violation}
              onChange={(e) => handleChange("violation", e.target.value)}
              className="px-3 py-3 border border-gray-700 rounded-lg bg-[#2A2B30] shadow-sm text-sm focus:outline-none focus:ring-2 focus:ring-[#5388DF] text-gray-200"
            >
              <option value="">All Violations</option>
              {violationOptions.map((vio) => (
                <option key={vio} value={vio}>
                  {vio}
                </option>
              ))}
            </select>
          </div>

          {/* Sort By */}
          <div className="flex flex-col w-40">
            <label className="font-medium text-sm mb-1 text-gray-400">
              Sort By
            </label>
            <select
              value={filters.sortBy}
              onChange={(e) => handleChange("sortBy", e.target.value)}
              className="px-3 py-3 border border-gray-700 rounded-lg bg-[#2A2B30] shadow-sm text-sm focus:outline-none focus:ring-2 focus:ring-[#5388DF] text-gray-200"
            >
              <option value="newest">Newest</option>
              <option value="oldest">Oldest</option>
            </select>
          </div>

          <div className="flex flex-col w-40">
            <label className="font-medium text-sm mb-1 text-gray-400">Date</label>
            <div className="relative">
              <DatePicker
                selected={filters.date}
                onChange={handleDateChange}
                dateFormat="yyyy/MM/dd"
                className="px-3 py-3 border border-gray-700 rounded-lg bg-[#2A2B30] shadow-sm text-sm focus:outline-none focus:ring-2 focus:ring-[#5388DF] text-gray-200 w-full pr-16"
                placeholderText="Select Date"
              />
              <FiCalendar className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-400 pointer-events-none" />
            </div>
          </div>
        </div>
      </section>

      {/* Incident List */}
      <section>
        <h2 className="text-xl font-semibold mb-4">Incident List</h2>
        <p className="text-gray-600 text-sm mb-6">
          All recorded incidents are listed below.
        </p>

        <div className="grid grid-cols-7 gap-4 mb-4 text-xs md:text-sm font-semibold text-white">
          <div className="bg-[#19325C] px-4 py-2 rounded-lg">Worker No.</div>
          <div className="bg-[#19325C] px-4 py-2 rounded-lg">Worker Name</div>
          <div className="bg-[#19325C] px-4 py-2 rounded-lg">Camera Location</div>
          <div className="bg-[#19325C] px-4 py-2 rounded-lg">Violation</div>
            <div className="bg-[#19325C] px-4 py-2 rounded-lg">Status</div>
          <div className="bg-[#19325C] px-4 py-2 rounded-lg">Date & Time</div>
          <div className="bg-[#19325C] px-4 py-2 rounded-lg text-center">
            Action
          </div>
        </div>

        {/* Status badge added */}
        <div className="space-y-3">
          {currentItems.map((n) => (
            <div
              key={n.id}
              className="grid grid-cols-7 gap-4 bg-[#2A2B30] rounded-lg shadow-sm border border-gray-700 p-4 hover:bg-[#3A3B40] transition-colors items-center"
            >
              <div className="font-medium text-gray-200">{n.worker_code}</div>
              <div className="text-gray-200">{n.worker}</div>
              <div className="text-gray-300">{n.camera || "Unknown Camera"}</div>

              <div className="text-gray-300">{n.violation}</div>
              <div>
                <span
                  className={`px-3 py-1 text-xs font-semibold rounded-full ${getStatusColor(
                    n.status
                  )}`}
                >
                  {n.status ? n.status.toUpperCase() : "No Status"}
                </span>
              </div>
              <div className="text-gray-300">
                 {n.created_at ? new Date(n.created_at).toLocaleString("en-US", {
                month: "short",
                day: "numeric",
                year: "numeric",
                hour: "2-digit",
                minute: "2-digit"
              }) : "-"}
              </div>
              <div className="text-center">
                 <button
                  onClick={() => handleView(n)}
                  className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-[#5388DF] rounded-md hover:bg-[#19325C] transition-colors"
                >
                  <FaEye className="mr-2" /> View
                </button>
              </div>
            </div>
          ))}
        </div>

        <div className="flex justify-center mt-6 space-x-2">
          <button
            onClick={() => setCurrentPage((prev) => Math.max(1, prev - 1))}
            disabled={currentPage === 1}
            className="px-4 py-2 bg-[#2A2B30] text-gray-200 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[#19325C] transition"
          >
            Previous
          </button>
          {Array.from({ length: totalPages }, (_, i) => (
            <button
              key={i + 1}
              onClick={() => setCurrentPage(i + 1)}
              className={`px-4 py-2 rounded-lg ${
                currentPage === i + 1
                  ? "bg-[#5388DF] text-white"
                  : "bg-[#2A2B30] text-gray-200 hover:bg-[#19325C]"
              } transition`}
            >
              {i + 1}
            </button>
          ))}
          <button
            onClick={() =>
              setCurrentPage((prev) => Math.min(totalPages, prev + 1))
            }
            disabled={currentPage === totalPages}
            className="px-4 py-2 bg-[#2A2B30] text-gray-200 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[#19325C] transition"
          >
            Next
          </button>
        </div>
      </section>

      {/* Modal */}
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
        if (!res.ok) throw new Error("Failed to update");
        const data = await res.json();

       
        setNotifications((prev) =>
          prev.map((v) =>
            v.id === selectedViolation.id ? { ...v, status: data.status } : v
          )
        );
        setSelectedViolation((prev) => ({ ...prev, status: data.status }));
      } catch (err) {
        console.error("Error updating status:", err);
      }
    }}
  />
)}

    </div>
  );
}

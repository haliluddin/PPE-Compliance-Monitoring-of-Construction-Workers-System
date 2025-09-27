import React, { useState } from "react";
import { FaEye, FaCheckCircle, FaExclamationCircle } from "react-icons/fa";

export default function Incident() {
  const notifications = [
    {
      id: 1,
      camera: "Camera A",
      violation: "No Helmet",
      status: "Pending",
      date: "2025-09-27 10:35 AM",
      worker: "Worker 1",
    },
    {
      id: 2,
      camera: "Camera B",
      violation: "No Vest",
      status: "Resolved",
      date: "2025-09-26 4:20 PM",
      worker: "Worker 2",
    },
  ];

  /* dropdown lists */
  const cameraOptions   = ["Camera A", "Camera B", "Camera C"];
  const violationOptions = ["No Helmet", "No Vest", "No Gloves"];
  const statusOptions    = ["Pending", "Resolved", "Under Review"];

  /* selected value */
  const [filters, setFilters] = useState({
    camera: "",
    violation: "",
    status: "",
  });

  const handleChange = (key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };
  const statusBadge = (status) => {
    switch (status) {
      case "Resolved":
        return (
          <span className="inline-flex items-center gap-1 px-3 py-1 text-xs font-medium rounded-full bg-green-100 text-green-700">
            <FaCheckCircle className="w-4 h-4" />
            {status}
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-3 py-1 text-xs font-medium rounded-full bg-yellow-100 text-yellow-700">
            <FaExclamationCircle className="w-4 h-4" />
            {status}
          </span>
        );
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 text-[#19325C] p-6">
      {/* ---------- Page Header ---------- */}
      <header className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Incident Records</h1>
        <p className="text-gray-600 max-w-2xl text-sm">
          Review and manage all safety violation incidents detected on site. 
          Use the filters below to narrow the list, and click <strong>View</strong> 
          to see full details of each incident.
        </p>
      </header>

      {/* ---------- Filters ---------- */}
      <section className="mb-10">
        <h2 className="text-xl font-semibold mb-2">Filter Incidents</h2>
        <p className="text-gray-600 text-sm mb-4">
          Select a camera, violation type, or status to filter results. 
        </p>

        <div className="flex flex-wrap gap-6">
          {/* Camera Dropdown */}
          <div className="flex flex-col w-60">
            <label className="font-medium text-sm mb-1">Camera</label>
            <select
              value={filters.camera}
              onChange={(e) => handleChange("camera", e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg bg-white shadow-sm text-sm focus:outline-none focus:ring-2 focus:ring-[#19325C]"
            >
              <option value="">Select a camera</option>
              {cameraOptions.map((cam) => (
                <option key={cam} value={cam}>{cam}</option>
              ))}
            </select>
            <span className="text-xs text-gray-500 mt-1">
              Pick a camera to filter results.
            </span>
          </div>

          {/* Violation Dropdown */}
          <div className="flex flex-col w-60">
            <label className="font-medium text-sm mb-1">Violation Type</label>
            <select
              value={filters.violation}
              onChange={(e) => handleChange("violation", e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg bg-white shadow-sm text-sm focus:outline-none focus:ring-2 focus:ring-[#19325C]"
            >
              <option value="">Select a violation</option>
              {violationOptions.map((vio) => (
                <option key={vio} value={vio}>{vio}</option>
              ))}
            </select>
            <span className="text-xs text-gray-500 mt-1">
              Pick a violation type to filter results.
            </span>
          </div>

          {/* Status Dropdown */}
          <div className="flex flex-col w-60">
            <label className="font-medium text-sm mb-1">Status</label>
            <select
              value={filters.status}
              onChange={(e) => handleChange("status", e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg bg-white shadow-sm text-sm focus:outline-none focus:ring-2 focus:ring-[#19325C]"
            >
              <option value="">Select a status</option>
              {statusOptions.map((stat) => (
                <option key={stat} value={stat}>{stat}</option>
              ))}
            </select>
            <span className="text-xs text-gray-500 mt-1">
              Pick a status to filter results.
            </span>
          </div>
        </div>
      </section>
      
      {/* ---------- Incident Table ---------- */}
      <section>
        <h2 className="text-xl font-semibold mb-4">Incident List</h2>
        <p className="text-gray-600 text-sm mb-6">
          All recorded incidents with their status and key details are listed below.
        </p>

        <div className="overflow-x-auto rounded-xl border border-gray-200 shadow-sm bg-white">
          <table className="min-w-full divide-y divide-gray-200 text-sm">
            <thead className="bg-[#19325C] text-white">
              <tr>
                <th className="px-6 py-3 text-left font-semibold tracking-wider">Worker</th>
                <th className="px-6 py-3 text-left font-semibold tracking-wider">Camera Location</th>
                <th className="px-6 py-3 text-left font-semibold tracking-wider">Violation</th>
                <th className="px-6 py-3 text-left font-semibold tracking-wider">Status</th>
                <th className="px-6 py-3 text-left font-semibold tracking-wider">Date & Time</th>
                <th className="px-6 py-3 text-center font-semibold tracking-wider">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {notifications.map((n) => (
                <tr
                  key={n.id}
                  className="hover:bg-gray-50 transition-colors duration-150"
                >
                  <td className="px-6 py-4 font-medium">{n.worker}</td>
                  <td className="px-6 py-4">{n.camera}</td>
                  <td className="px-6 py-4">{n.violation}</td>
                  <td className="px-6 py-4">{statusBadge(n.status)}</td>
                  <td className="px-6 py-4">{n.date}</td>
                  <td className="px-6 py-4 text-center">
                    <button
                      className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-[#19325C] rounded-lg hover:bg-[#152747] transition-colors"
                    >
                      <FaEye className="mr-2" />
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

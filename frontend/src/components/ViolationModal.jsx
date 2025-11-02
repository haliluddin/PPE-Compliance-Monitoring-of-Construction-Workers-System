import React, { useState } from "react";
import { FiX, FiUser, FiCamera, FiAlertTriangle, FiClock } from "react-icons/fi";

export default function ViolationModal({ violation, onClose, onStatusChange }) {
  const [updating, setUpdating] = useState(false);

  const handleStatusChange = async (e) => {
    const newStatus = e.target.value;
    setUpdating(true);
    try {
      await onStatusChange(newStatus);
    } finally {
      setUpdating(false);
    }
  };

  if (!violation) return null;

  const snapshotUrl = violation.snapshot
    ? (typeof violation.snapshot === "string" ? `data:image/jpeg;base64,${violation.snapshot}` : `data:image/jpeg;base64,${violation.snapshot}`)
    : "https://via.placeholder.com/400x250?text=No+Snapshot+Available";

  const getWorkerName = (v) => {
    const n = v.worker || v.worker_name || v.worker_name_obj || v.worker_code;
    if (!n) return "N/A";
    if (typeof n === "string") return n;
    if (typeof n === "object") {
      return n.fullName || n.name || n.firstName || `${n.first || ""} ${n.last || ""}`.trim() || JSON.stringify(n);
    }
    return String(n);
  };

  let dateStr = "N/A";
  try {
    dateStr = violation.created_at || violation.date ? new Date(violation.created_at || violation.date).toLocaleString() : "N/A";
  } catch (e) {
    dateStr = String(violation.created_at || violation.date || "N/A");
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-70 flex justify-center items-center z-50 p-4">
      <div className="bg-[#1F2025] rounded-2xl shadow-2xl w-full max-w-4xl relative overflow-hidden">
        <div className="flex justify-between items-center px-6 py-4 border-b border-gray-700">
          <h2 className="text-xl font-semibold text-white">Violation Details</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition"
          >
            <FiX size={22} />
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-6">
          <div className="flex flex-col">
            <div className="bg-[#2A2B30] rounded-xl overflow-hidden border border-gray-700 shadow-inner mb-4">
              <img
                src={snapshotUrl}
                alt="Violation Snapshot"
                className="w-full h-64 object-cover"
              />
            </div>

            <div className="bg-[#2A2B30] rounded-xl p-4 border border-gray-700">
              <label className="block text-sm text-gray-400 mb-2 font-medium">
                Update Status
              </label>
              <select
                value={violation.status}
                onChange={handleStatusChange}
                disabled={updating}
                className="w-full px-3 py-2 rounded-lg bg-[#1E1F23] border border-gray-600 text-gray-200 focus:ring-2 focus:ring-[#5388DF] transition"
              >
                <option value="pending">Pending</option>
                <option value="resolved">Resolved</option>
                <option value="false positive">False Positive</option>
              </select>
              {updating && (
                <p className="text-xs text-blue-400 mt-2">Updating status...</p>
              )}
            </div>
          </div>

          <div className="flex flex-col justify-between bg-[#2A2B30] rounded-xl p-5 border border-gray-700">
            <div className="space-y-4">
              <div className="flex items-center text-gray-300">
                <FiUser className="mr-2 text-[#5388DF]" />
                <p>
                  <span className="text-gray-400 font-medium">Worker:</span>{" "}
                  {getWorkerName(violation)}
                </p>
              </div>

              <div className="flex items-center text-gray-300">
                <FiCamera className="mr-2 text-[#5388DF]" />
                <p>
                  <span className="text-gray-400 font-medium">Camera:</span>{" "}
                  {violation.camera || violation.cameraLocation || "N/A"}
                </p>
              </div>

              <div className="flex items-center text-gray-300">
                <FiAlertTriangle className="mr-2 text-[#5388DF]" />
                <p>
                  <span className="text-gray-400 font-medium">Violation:</span>{" "}
                  {violation.violation || violation.type || "N/A"}
                </p>
              </div>

              <div className="flex items-center text-gray-300">
                <FiClock className="mr-2 text-[#5388DF]" />
                <p>
                  <span className="text-gray-400 font-medium">Date & Time:</span>{" "}
                  {dateStr}
                </p>
              </div>
            </div>

            <div className="mt-6 text-right">
              <button
                onClick={onClose}
                className="bg-[#5388DF] hover:bg-[#19325C] text-white px-5 py-2 rounded-lg transition"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

import React, { useState } from "react";
import { FiX, FiUser, FiCamera, FiAlertTriangle, FiClock } from "react-icons/fi";
import API from "../api";

export default function ViolationModal({ violation, onClose, onStatusChange }) {
  const [updating, setUpdating] = useState(false);

  const handleStatusChange = async (event) => {
    const newStatus = event.target.value;
    setUpdating(true);

    try {
      const response = await API.put(`/violations/${violation.violation_id}/status`, {
        status: newStatus.toLowerCase()
      });

      if (response.status === 200 || response.status === 204) {
        // call parent's handler if available — be flexible about the expected args
        if (onStatusChange) {
          try {
            if (typeof onStatusChange === "function") {
              // call with (violation_id, newStatus) if the function expects 2 args,
              // otherwise call with single newStatus for backward compatibility.
              if (onStatusChange.length >= 2) {
                onStatusChange(violation.violation_id, newStatus);
              } else {
                onStatusChange(newStatus);
              }
            }
          } catch (e) {
            // swallow parent handler errors — we still close modal
            console.warn("onStatusChange handler errored:", e);
          }
        }
        // Close modal (parent should refresh relevant lists/stats)
        if (onClose) onClose();
      } else {
        // If backend responded non-OK, revert select visually
        try { event.target.value = violation.status; } catch (e) {}
      }
    } catch (error) {
      console.error("Status update failed:", error);
      try { event.target.value = violation.status; } catch (e) {}
    } finally {
      setUpdating(false);
    }
  };

  if (!violation) return null;

  const snapshotVal = violation.snapshot || violation.snapshot_base64 || violation.snapshot_b64;
  const snapshotUrl = snapshotVal 
    ? (snapshotVal.startsWith('data:image') ? snapshotVal : `data:image/jpeg;base64,${snapshotVal}`)
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

  const getCameraDisplay = (v) => {
    return v.camera || v.camera_location || v.cameraLocation || v.cameraLocationName || "N/A";
  };

  const getViolationType = (v) => {
    return v.violation || v.violation_type || v.violation_types || v.type || "N/A";
  };

  const formatPhTime = (dateStr) => {
    try {
      const date = new Date(dateStr);
      return date.toLocaleString('en-PH', {
        timeZone: 'Asia/Manila',
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: 'numeric',
        minute: 'numeric',
        hour12: true
      });
    } catch (e) {
      return 'N/A';
    }
  };

  dateStr = formatPhTime(violation.created_at);

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
                  {getCameraDisplay(violation)}
                </p>
              </div>

              <div className="flex items-center text-gray-300">
                <FiAlertTriangle className="mr-2 text-[#5388DF]" />
                <p>
                  <span className="text-gray-400 font-medium">Violation:</span>{" "}
                  {getViolationType(violation)}
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

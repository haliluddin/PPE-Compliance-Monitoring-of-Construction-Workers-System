import React, { useState } from "react";

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

  // Convert snapshot (binary base64) to usable image URL
  const snapshotUrl = violation.snapshot
    ? `data:image/jpeg;base64,${violation.snapshot}`
    : "https://via.placeholder.com/400x250?text=No+Snapshot+Available"; // ✅ placeholder

  return (
    <div className="fixed inset-0 bg-black bg-opacity-60 flex justify-center items-center z-50">
      <div className="bg-[#2A2B30] rounded-xl p-6 w-[420px] shadow-lg relative">
        <h2 className="text-lg font-semibold mb-4">Violation Details</h2>

        {/* ✅ Snapshot Section */}
        <div className="mb-4">
          <img
            src={snapshotUrl}
            alt="Violation Snapshot"
            className="w-full h-56 object-cover rounded-lg border border-gray-700"
          />
        </div>

        <p><strong>Worker:</strong> {violation.worker}</p>
        <p><strong>Camera:</strong> {violation.camera}</p>
        <p><strong>Violation:</strong> {violation.violation}</p>
        <p>
          <strong>Date & Time:</strong>{" "}
          {new Date(violation.frame_ts).toLocaleString()}
        </p>

        <div className="mt-4">
          <label className="block text-sm text-gray-400 mb-1">
            Update Status
          </label>
          <select
            value={violation.status}
            onChange={handleStatusChange}
            disabled={updating}
            className="w-full px-3 py-2 rounded-lg bg-[#1E1F23] border border-gray-600 text-gray-200 focus:ring-2 focus:ring-[#5388DF]"
          >
            <option value="pending">Pending</option>
            <option value="resolved">Resolved</option>
            <option value="false positive">False Positive</option>
          </select>
        </div>

        <button
          onClick={onClose}
          className="mt-6 bg-[#5388DF] hover:bg-[#19325C] text-white px-4 py-2 rounded-lg w-full"
        >
          Close
        </button>
      </div>
    </div>
  );
}

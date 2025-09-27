import { FaUserAlt, FaEye, FaPlus, FaSearch } from "react-icons/fa";
import { useState } from "react";

export default function Workers() {
  const [query, setQuery] = useState("");

  const workers = [
    { id: "1", name: "Juan Dela Cruz", lastSeen: "2025-09-27 10:45 AM", totalIncidents: 3 },
    { id: "2", name: "Maria Santos", lastSeen: "2025-09-27 9:20 AM", totalIncidents: 0 },
    { id: "3", name: "Carlos Reyes", lastSeen: "2025-09-26 5:10 PM", totalIncidents: 1 },
  ];

  const filtered = workers.filter(
    (w) =>
      w.name.toLowerCase().includes(query.toLowerCase()) ||
      w.id.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-gray-50 text-[#19325C] p-6">
      {/* ---------- Page Header ---------- */}
      <header className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Workers Profiling</h1>
        <p className="text-gray-600 max-w-2xl text-sm">
          Track worker activity, last seen time, and incident history.
        </p>
      </header>

      {/* ---------- Action Bar ---------- */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6 gap-3">
        {/* Search box */}
        <div className="relative w-full sm:w-1/3">
          <FaSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search by ID or Name"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg shadow-sm text-sm
                       focus:outline-none focus:ring-2 focus:ring-[#19325C]"
          />
        </div>

        {/* Add Worker Button */}
        <button
          className="inline-flex items-center justify-center px-5 py-2 text-sm font-medium text-white
                     bg-[#19325C] rounded-lg hover:bg-[#152747] transition-colors"
          onClick={() => alert("Add Worker form goes here")}
        >
          <FaPlus className="mr-2" />
          Add Worker
        </button>
      </div>
      
      <div className="grid grid-cols-5 gap-4 mb-4 text-xs md:text-sm font-semibold text-white">
        <div className="bg-[#19325C] px-4 py-2 rounded-lg">Worker ID</div>
        <div className="bg-[#19325C] px-4 py-2 rounded-lg">Name</div>
        <div className="bg-[#19325C] px-4 py-2 rounded-lg">Last Seen</div>
        <div className="bg-[#19325C] px-4 py-2 rounded-lg">Total Incidents</div>
        <div className="bg-[#19325C] px-4 py-2 rounded-lg text-center">Action</div>
      </div>

      {/* ---------- Cards ---------- */}
      <div className="space-y-3">
        {filtered.map((w) => (
          <div
            key={w.id}
            className="grid grid-cols-5 gap-4 bg-white rounded-xl shadow-sm border border-gray-200 p-4 hover:shadow-md transition-shadow"
          >
            <div className="flex items-center gap-2 font-medium">
              <FaUserAlt className="text-[#19325C]" />
              {w.id}
            </div>

            <div>{w.name}</div>
            <div>{w.lastSeen}</div>

            <div>
              {w.totalIncidents === 0 ? (
                <span className="text-green-600 font-medium">0</span>
              ) : (
                <span className="text-red-600 font-semibold">
                  {w.totalIncidents}
                </span>
              )}
            </div>

            <div className="text-center">
              <button className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-[#19325C] rounded-lg hover:bg-[#152747] transition-colors">
                <FaEye className="mr-2" />
                View Profile
              </button>
            </div>
          </div>
        ))}

        {filtered.length === 0 && (
          <p className="text-gray-500 text-sm mt-4">No workers found.</p>
        )}
      </div>
    </div>
  );
}

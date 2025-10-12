// frontend/src/pages/Workers.jsx
import { FaUserAlt, FaEye, FaPlus, FaSearch, FaTimes } from "react-icons/fa";
import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import API from "../api";

export default function Workers() {
  const [query, setQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(5);
  const [workers, setWorkers] = useState([]);
  const [isAddWorkerModalOpen, setIsAddWorkerModalOpen] = useState(false);
  const [newWorker, setNewWorker] = useState({
    fullName: "",
    worker_code: "",
    assignedLocation: "",
    role: "",
    dateAdded: new Date().toISOString().split("T")[0],
    status: "Active",
  });

  // Fetch workers from backend
  const fetchWorkers = async () => {
    try {
      const res = await API.get("/workers");
      setWorkers(res.data);
    } catch (error) {
      console.error("Failed to fetch workers:", error);
    }
  };

  useEffect(() => {
    fetchWorkers();
  }, []);

  // Handle form input
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewWorker((prev) => ({ ...prev, [name]: value }));
  };

  // Add new worker
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await API.post("/workers", newWorker);
      setWorkers((prev) => [...prev, res.data]);
      setNewWorker({
        fullName: "",
        worker_code: "",
        assignedLocation: "",
        role: "",
        dateAdded: new Date().toISOString().split("T")[0],
        status: "Active",
      });
      setIsAddWorkerModalOpen(false);
    } catch (error) {
      console.error("Failed to add worker:", error);
      alert(error.response?.data?.detail || "Failed to add worker");
    }
  };

  // Filter & Pagination
  const filtered = workers.filter(
    (w) =>
      w.fullName.toLowerCase().includes(query.toLowerCase()) ||
      w.worker_code.toLowerCase().includes(query.toLowerCase())
  );

  const totalPages = Math.ceil(filtered.length / itemsPerPage);
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentItems = filtered.slice(indexOfFirstItem, indexOfLastItem);

  return (
    <div className="min-h-screen bg-[#1E1F23] text-gray-100 p-6">
      {/* Page Header */}
      <header className="bg-[#2A2B30] px-5 py-3 rounded-xl shadow-lg mb-8 flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-100">Worker Management</h1>
        <div className="flex items-center space-x-2 text-gray-300">
          <span className="text-lg font-semibold">Total Workers:</span>
          <span className="text-xl font-bold text-[#5388DF]">{workers.length}</span>
        </div>
      </header>

      {/* Action Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6 gap-3">
        {/* Search */}
        <div className="relative w-64">
          <FaSearch className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search by ID or Name"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full pl-12 pr-4 py-3 border border-gray-700 rounded-lg shadow-sm text-sm focus:outline-none focus:ring-2 focus:ring-[#5388DF] bg-[#2A2B30] text-gray-200"
          />
        </div>

        {/* Add Worker Button */}
        <button
          className="inline-flex items-center justify-center px-5 py-3 text-sm font-medium text-white bg-[#5388DF] rounded-lg hover:bg-[#19325C] transition-colors"
          onClick={() => setIsAddWorkerModalOpen(true)}
        >
          <FaPlus className="mr-2" />
          Add Worker
        </button>
      </div>

      {/* Add Worker Modal */}
      {isAddWorkerModalOpen && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="bg-[#1E1F21] rounded-xl w-full max-w-2xl border border-gray-700 shadow-2xl overflow-hidden">
            <div className="px-6 py-5 border-b border-gray-700 flex justify-between items-center">
              <div>
                <h3 className="text-xl font-semibold text-white">Add New Worker</h3>
                <p className="text-sm text-gray-400 mt-1">Fill in the worker's details below</p>
              </div>
              <button
                onClick={() => setIsAddWorkerModalOpen(false)}
                className="text-gray-400 hover:text-white p-1 rounded-full hover:bg-gray-700/50 transition-colors"
              >
                <FaTimes className="h-5 w-5" />
              </button>
            </div>

            <form onSubmit={handleSubmit} className="p-6 space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Personal Info */}
                <div className="space-y-5">
                  <div className="space-y-1">
                    <label className="block text-sm font-medium text-gray-300">Full Name</label>
                    <input
                      type="text"
                      name="fullName"
                      value={newWorker.fullName}
                      onChange={handleInputChange}
                      placeholder="Juan Dela Cruz"
                      className="w-full px-4 py-2.5 bg-[#2A2B30] border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-[#5388DF]"
                      required
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="block text-sm font-medium text-gray-300">Worker Code</label>
                    <input
                      type="text"
                      name="worker_code"
                      value={newWorker.worker_code}
                      onChange={handleInputChange}
                      placeholder="[Vest No] 1"
                      className="w-full px-4 py-2.5 bg-[#2A2B30] border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-[#5388DF]"
                      required
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="block text-sm font-medium text-gray-300">Assigned Location</label>
                    <input
                      type="text"
                      name="assignedLocation"
                      value={newWorker.assignedLocation}
                      onChange={handleInputChange}
                      placeholder="Camera 5 – Ground Floor"
                      className="w-full px-4 py-2.5 bg-[#2A2B30] border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-[#5388DF]"
                      required
                    />
                  </div>
                </div>

                {/* Work Info */}
                <div className="space-y-5">
                  <div className="space-y-1">
                    <label className="block text-sm font-medium text-gray-300">Role</label>
                    <select
                      name="role"
                      value={newWorker.role}
                      onChange={handleInputChange}
                      className="w-full px-4 py-2.5 bg-[#2A2B30] border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-[#5388DF] appearance-none"
                      required
                    >
                      <option value="">Select Role</option>
                      <option value="Welder">Welder</option>
                      <option value="Operator">Operator</option>
                      <option value="Carpenter">Carpenter</option>
                      <option value="Electrician">Electrician</option>
                      <option value="Plumber">Plumber</option>
                      <option value="Laborer">Laborer</option>
                      <option value="Foreman">Foreman</option>
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="block text-sm font-medium text-gray-300">Date Added</label>
                    <input
                      type="date"
                      name="dateAdded"
                      value={newWorker.dateAdded}
                      onChange={handleInputChange}
                      className="w-full px-4 py-2.5 bg-[#2A2B30] border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-[#5388DF]"
                      required
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="block text-sm font-medium text-gray-300">Status</label>
                    <select
                      name="status"
                      value={newWorker.status}
                      onChange={handleInputChange}
                      className="w-full px-4 py-2.5 bg-[#2A2B30] border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-[#5388DF] appearance-none"
                      required
                    >
                      <option value="Active">Active</option>
                      <option value="Inactive">Inactive</option>
                      <option value="On Leave">On Leave</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Form Actions */}
              <div className="flex justify-end space-x-3 pt-4">
                <button
                  type="button"
                  onClick={() => setIsAddWorkerModalOpen(false)}
                  className="px-5 py-2.5 text-sm font-medium text-gray-300 bg-transparent border border-gray-600 rounded-lg hover:bg-gray-700/50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-5 py-2.5 text-sm font-medium text-white bg-[#5388DF] rounded-lg hover:bg-[#3a6fc5] flex items-center justify-center"
                >
                  <FaPlus className="mr-2" />
                  Add Worker
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Workers Table */}
      <div className="grid grid-cols-5 gap-4 mb-4 text-xs md:text-sm font-semibold text-white">
        <div className="bg-[#19325C] px-4 py-2 rounded-lg">Worker Code</div>
        <div className="bg-[#19325C] px-4 py-2 rounded-lg">Name</div>
        <div className="bg-[#19325C] px-4 py-2 rounded-lg">Last Seen</div>
        <div className="bg-[#19325C] px-4 py-2 rounded-lg">Total Incidents</div>
        <div className="bg-[#19325C] px-4 py-2 rounded-lg text-center">Action</div>
      </div>

      <div className="space-y-3">
        {currentItems.map((w) => (
          <div
            key={w.id}
            className="grid grid-cols-5 gap-4 bg-[#2A2B30] rounded-lg shadow-sm border border-gray-700 p-4 hover:bg-[#3A3B40] transition-shadow items-center"
          >
            <div className="flex items-center gap-2 font-medium text-gray-200">
              <FaUserAlt className="text-[#5388DF]" />
              {w.worker_code}
            </div>
            <div className="text-gray-200">{w.fullName}</div>
            <div className="text-gray-300">{w.lastSeen || "-"}</div>
            <div>{w.totalIncidents || 0}</div>
            <div className="text-center">
              <Link
                to={`/workersprofile/${w.id}`}
                className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-[#5388DF] rounded-md hover:bg-[#19325C]"
              >
                <FaEye className="mr-2" />
                View Profile
              </Link>
            </div>
          </div>
        ))}
        {filtered.length === 0 && <p className="text-gray-500 text-sm mt-4">No workers found.</p>}
      </div>

      {/* Pagination */}
      <div className="flex justify-center mt-6 space-x-2">
        <button
          onClick={() => setCurrentPage((prev) => Math.max(1, prev - 1))}
          disabled={currentPage === 1}
          className="px-4 py-2 bg-[#2A2B30] text-gray-200 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[#19325C]"
        >
          Previous
        </button>
        {Array.from({ length: totalPages }, (_, i) => (
          <button
            key={i + 1}
            onClick={() => setCurrentPage(i + 1)}
            className={`px-4 py-2 rounded-lg ${
              currentPage === i + 1 ? "bg-[#5388DF] text-white" : "bg-[#2A2B30] text-gray-200 hover:bg-[#19325C]"
            }`}
          >
            {i + 1}
          </button>
        ))}
        <button
          onClick={() => setCurrentPage((prev) => Math.min(totalPages, prev + 1))}
          disabled={currentPage === totalPages}
          className="px-4 py-2 bg-[#2A2B30] text-gray-200 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[#19325C]"
        >
          Next
        </button>
      </div>
    </div>
  );
}

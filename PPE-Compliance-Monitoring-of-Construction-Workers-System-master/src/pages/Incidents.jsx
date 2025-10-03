import React, { useState } from "react";
import { FaEye } from "react-icons/fa";
import { FiSearch, FiCalendar } from "react-icons/fi";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import { format } from "date-fns";


export default function Incident() {
  const notifications = [
    { id: 1, camera: "Camera 2", violation: "No Helmet",  date: "2025-09-27 10:35:00", worker: "John Doe", workerNo: "1" },
    { id: 2, camera: "Camera 3", violation: "No Vest",    date: "2025-09-26 16:20:00",  worker: "Jane Smith", workerNo: "2" },
    { id: 3, camera: "Camera 3", violation: "No Gloves",  date: "2025-09-27 09:00:00", worker: "Peter Jones", workerNo: "3" },
    { id: 4, camera: "Camera 3", violation: "No Helmet",  date: "2025-09-25 11:15:00", worker: "Alice Brown", workerNo: "4" },
    { id: 5, camera: "Camera 5", violation: "No Safety Shoes", date: "2025-09-24 14:30:00", worker: "Robert Green", workerNo: "5" },
    { id: 6, camera: "Camera 7", violation: "No Vest",    date: "2025-09-23 13:00:00", worker: "Sarah White", workerNo: "6" },
    { id: 7, camera: "Camera 10", violation: "No Gloves",  date: "2025-09-22 08:45:00", worker: "David Black", workerNo: "7" },
  ];

  const cameraOptions   = ["Camera 1", "Camera 2", "Camera 3"];
  const violationOptions = ["No Helmet", "No Vest", "No Gloves", "No Safety Shoes"];
  

  const [filters, setFilters] = useState({ camera: "", violation: "", date: null, sortBy: "newest" });
  const handleChange = (key, value) => setFilters((p) => ({ ...p, [key]: value }));
  const handleDateChange = (date) => handleChange("date", date);

  const [searchQuery, setSearchQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(5); 

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

      if (filters.camera && n.camera !== filters.camera) return false;
      if (filters.violation && n.violation !== filters.violation) return false;
      if (filters.date) {
        const notificationDate = new Date(n.date.split(' ')[0]); // Get only the date part for comparison
        const filterDate = new Date(filters.date);
        if (notificationDate.toDateString() !== filterDate.toDateString()) return false;
      }
      return true;
    })
    .sort((a, b) => {
      const dateA = new Date(a.date);
      const dateB = new Date(b.date);
      if (filters.sortBy === "newest") {
        return dateB.getTime() - dateA.getTime(); // Newest first
      } else {
        return dateA.getTime() - dateB.getTime(); // Oldest first
      }
    });

  const totalPages = Math.ceil(filteredNotifications.length / itemsPerPage);
  const indexOfLastItem = currentPage * itemsPerPage;
  const indexOfFirstItem = indexOfLastItem - itemsPerPage;
  const currentItems = filteredNotifications.slice(indexOfFirstItem, indexOfLastItem);

  return (
    <div className="min-h-screen bg-[#1E1F23] text-gray-100 p-6">
      {/* ---------- Page Header ---------- */}
      <div className="bg-[#2A2B30] px-5 py-3 rounded-xl shadow-lg mb-8 flex items-center justify-between">
        {/* <h1 className="text-2xl font-bold text-gray-100">Incident Records</h1> */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 text-gray-300">
            <span className="text-lg font-semibold">Total Incidents:</span>
            <span className="text-xl font-bold text-[#5388DF]">{notifications.length}</span>
          </div>
        </div>
      </div>

    

      {/* ---------- Filters ---------- */}
      <section className="mb-10">
        <div className="flex flex-col md:flex-row items-stretch md:items-center gap-6">
          {/* Search Bar */}
          <div className="relative w-full md:w-auto flex-1">
            <label htmlFor="search-incidents" className="sr-only">Search Incidents</label>
            <input
              type="text"
              id="search-incidents"
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
            <label className="font-medium text-sm mb-1 text-gray-400">Camera Location</label>
            <select
              value={filters.camera}
              onChange={(e) => handleChange("camera", e.target.value)}
              className="px-3 py-3 border border-gray-700 rounded-lg bg-[#2A2B30] shadow-sm text-sm focus:outline-none focus:ring-2 focus:ring-[#5388DF] text-gray-200"
            >
              <option value="">All Camera</option>
              {cameraOptions.map((cam) => (
                <option key={cam} value={cam}>{cam}</option>
              ))}
            </select>
          </div>

          {/* Violation */}
          <div className="flex flex-col w-40">
            <label className="font-medium text-sm mb-1 text-gray-400">Violation Type</label>
            <select
              value={filters.violation}
              onChange={(e) => handleChange("violation", e.target.value)}
              className="px-3 py-3 border border-gray-700 rounded-lg bg-[#2A2B30] shadow-sm text-sm focus:outline-none focus:ring-2 focus:ring-[#5388DF] text-gray-200"
            >
              <option value="">All Violations</option>
              {violationOptions.map((vio) => (
                <option key={vio} value={vio}>{vio}</option>
              ))}
            </select>
          </div>
          
          {/* Sort By */}
          <div className="flex flex-col w-40">
            <label className="font-medium text-sm mb-1 text-gray-400">Sort By</label>
            <select
              value={filters.sortBy}
              onChange={(e) => handleChange("sortBy", e.target.value)}
              className="px-3 py-3 border border-gray-700 rounded-lg bg-[#2A2B30] shadow-sm text-sm focus:outline-none focus:ring-2 focus:ring-[#5388DF] text-gray-200"
            >
              <option value="newest">Newest</option>
              <option value="oldest">Oldest</option>
            </select>
          </div>

          {/* Date Picker */}
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

      <section>
        <h2 className="text-xl font-semibold mb-4">Incident List</h2>
        <p className="text-gray-600 text-sm mb-6">
          All recorded incidents with their status and key details are listed below.
        </p>

        <div className="grid grid-cols-6 gap-4 mb-4 text-xs md:text-sm font-semibold text-white">
          <div className="bg-[#19325C] px-4 py-2 rounded-lg">Worker No.</div>
          <div className="bg-[#19325C] px-4 py-2 rounded-lg">Worker Name</div>
          <div className="bg-[#19325C] px-4 py-2 rounded-lg">Camera Location</div>
          <div className="bg-[#19325C] px-4 py-2 rounded-lg">Violation</div>
          <div className="bg-[#19325C] px-4 py-2 rounded-lg">Date & Time</div>
          <div className="bg-[#19325C] px-4 py-2 rounded-lg text-center">Action</div>
        </div>
      
        <div className="space-y-3">
          {currentItems.map((n) => (
            <div
              key={n.id}
              className="grid grid-cols-6 gap-4 bg-[#2A2B30] rounded-lg shadow-sm border border-gray-700 p-4 hover:bg-[#3A3B40] transition-colors items-center"
            >
              <div className="font-medium text-gray-200">{n.workerNo}</div>
              <div className="text-gray-200">{n.worker}</div>
              <div className="text-gray-300">{n.camera}</div>
              <div className="text-gray-300">{n.violation}</div>
              <div className="text-gray-300">{format(new Date(n.date), 'yyyy-MM-dd HH:mm:ss')}</div>
              <div className="text-center">
                <button className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-[#5388DF] rounded-md hover:bg-[#19325C] transition-colors">
                  <FaEye className="mr-2" />
                  View
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
              className={`px-4 py-2 rounded-lg ${currentPage === i + 1 ? 'bg-[#5388DF] text-white' : 'bg-[#2A2B30] text-gray-200 hover:bg-[#19325C]'} transition`}
            >
              {i + 1}
            </button>
          ))}
          <button
            onClick={() => setCurrentPage((prev) => Math.min(totalPages, prev + 1))}
            disabled={currentPage === totalPages}
            className="px-4 py-2 bg-[#2A2B30] text-gray-200 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[#19325C] transition"
          >
            Next
          </button>
        </div>
      </section>
    </div>
  );
}

// frontend/src/pages/WorkersProfile.jsx
import React, { useState, useEffect } from 'react';
import { ArrowLeft, ShieldCheck, Calendar } from 'lucide-react';
import { format, parseISO } from 'date-fns';
import { FaEye } from "react-icons/fa";
import { useParams, useNavigate } from 'react-router-dom';
import API from "../api";

export default function WorkersProfile() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [workerData, setWorkerData] = useState(null);
  const [isDetailsExpanded, setIsDetailsExpanded] = useState(false);
  const [showFilter, setShowFilter] = useState(false);
  const [dateRange, setDateRange] = useState({ start: '', end: '' });
  const [selectedType, setSelectedType] = useState('All');
  const [searchQuery, setSearchQuery] = useState('');

  // Fetch worker profile
  useEffect(() => {
    const fetchWorker = async () => {
      try {
        const res = await API.get(`/workers/${id}`);
        const data = res.data;

        // Compute compliance rate (example logic: compliance = % of 'No violation' records)
        const total = data.violationHistory?.length || 0;
        const compliant = data.violationHistory?.filter(v => v.type === 'No Violation').length || 0;
        data.complianceRate = total > 0 ? Math.round((compliant / total) * 100) : 100;

        // Sort violations descending by date for last violation
        const sortedViolations = data.violationHistory?.sort((a, b) => new Date(b.date) - new Date(a.date));
        data.lastViolationDate = sortedViolations?.[0]?.date || null;

        // Set total violations
        data.totalViolations = total;

        setWorkerData(data);
      } catch (error) {
        console.error("Failed to fetch worker profile:", error);
        setWorkerData(null);
      }
    };

    fetchWorker();
  }, [id]);

  // Filter violations safely
  const filterViolations = () => {
    if (!workerData?.violationHistory) return [];
    return workerData.violationHistory.filter(violation => {
      const violationDate = new Date(violation.date);
      const startDate = dateRange.start ? new Date(dateRange.start) : null;
      const endDate = dateRange.end ? new Date(dateRange.end + 'T23:59:59') : null;

      const matchesDate = (!startDate || violationDate >= startDate) &&
                          (!endDate || violationDate <= endDate);
      const matchesType = selectedType === 'All' || violation.type === selectedType;
      const matchesSearch = !searchQuery ||
        Object.values(violation).some(
          val => String(val).toLowerCase().includes(searchQuery.toLowerCase())
        );

      return matchesDate && matchesType && matchesSearch;
    });
  };

  // Close filter when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      const filterButton = event.target.closest('.filter-button');
      const filterDropdown = event.target.closest('.filter-dropdown');
      if (!filterButton && !filterDropdown) setShowFilter(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  if (!workerData) {
    return (
      <div className="min-h-screen bg-[#1E1F23] text-gray-100 p-6">
        <p className="text-gray-400">Loading worker profile...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#1E1F23] text-gray-100 p-6">
      {/* Back Button */}
      <button
        onClick={() => navigate('/workers')}
        className="mb-6 flex items-center text-[#5388DF] hover:text-[#19325C] transition-colors"
      >
        <ArrowLeft className="w-5 h-5 mr-2" />
        Back to Workers
      </button>

      {/* Worker Profile Header */}
      <div className="bg-[#2A2B30] p-6 mb-8 shadow-lg rounded-xl border border-gray-700">
        <div className="flex flex-col">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              {/* Profile Picture */}
              <div className="w-16 h-16 md:w-20 md:h-20 rounded-full bg-gray-700/50 border-2 border-gray-600 flex items-center justify-center flex-shrink-0">
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  className="h-8 w-8 md:h-10 md:w-10 text-gray-400" 
                  viewBox="0 0 24 24" 
                  fill="none" 
                  stroke="currentColor" 
                  strokeWidth="1.5" 
                  strokeLinecap="round" 
                  strokeLinejoin="round"
                >
                  <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                  <circle cx="12" cy="7" r="4"></circle>
                </svg>
              </div>

              {/* Name and Basic Info */}
              <div>
                <div className="flex items-center space-x-3">
                  <h1 className="text-xl md:text-2xl font-bold text-white">{workerData.fullName || workerData.name}</h1>
                  <span className={`px-2.5 py-1 text-xs font-medium rounded-full ${
                    workerData.status === 'Active' ? 'bg-green-900/30 text-green-300 border border-green-800/50' :
                    workerData.status === 'Inactive' ? 'bg-red-900/30 text-red-300 border border-red-800/50' :
                    'bg-yellow-900/30 text-yellow-300 border border-yellow-800/50'
                  }`}>
                    {workerData.status || 'Active'}
                  </span>
                </div>
                <div className="flex items-center mt-1.5 space-x-2">
                  <span className="text-sm text-gray-400">Worker No:</span>
                  <span className="text-sm text-blue-300 font-medium">{workerData.id}</span>
                </div>
              </div>
            </div>

            {/* Toggle Button */}
            <button 
              onClick={() => setIsDetailsExpanded(!isDetailsExpanded)}
              className="text-gray-400 hover:text-white transition-colors p-2 -mr-2"
              aria-label={isDetailsExpanded ? 'Hide details' : 'Show details'}
            >
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                className={`h-5 w-5 transform transition-transform duration-200 ${isDetailsExpanded ? 'rotate-180' : ''}`} 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          </div>

          {/* Collapsible Details */}
          {isDetailsExpanded && (
            <div className="mt-6 pt-6 border-t border-gray-700">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="space-y-1">
                  <p className="text-xs text-gray-400 font-medium uppercase tracking-wider">Worker Role</p>
                  <p className="text-white font-medium">{workerData.role || 'Not specified'}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-gray-400 font-medium uppercase tracking-wider">Assigned Location</p>
                  <p className="text-white font-medium">{workerData.assignedLocation || 'Not assigned'}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-gray-400 font-medium uppercase tracking-wider">Date Added</p>
                  <p className="text-white font-medium">
                    {workerData.dateAdded ? new Date(workerData.dateAdded).toLocaleDateString('en-US', {
                      year: 'numeric',
                      month: 'short',
                      day: 'numeric'
                    }) : 'N/A'}
                  </p>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-gray-400 font-medium uppercase tracking-wider">Last Active</p>
                  <p className="text-white font-medium">
                    {workerData.lastActive ? new Date(workerData.lastActive).toLocaleString() : 'N/A'}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Violation Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        {/* Total Violations Card */}
        <div className="bg-[#2A2B30] p-4 rounded-xl shadow-lg">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <p className="text-sm text-gray-400">Total Violations</p>
              <p className="text-2xl font-bold text-white mt-1">{workerData.totalViolations || 0}</p>
              <div className="mt-2">
                <span className={`text-xs font-medium ${
                  workerData.totalViolations > 10 ? 'text-red-400' : 'text-green-400'
                }`}>
                  {workerData.totalViolations > 10 ? 'Needs Attention' : 'Within Limits'}
                </span>
              </div>
            </div>
            <div className="p-4 bg-red-500/10 rounded-lg flex items-center justify-center ml-6">
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                className="h-8 w-8 text-red-400"
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" 
                />
              </svg>
            </div>
          </div>
        </div>

        {/* Compliance Rate Card */}
        <div className="bg-[#2A2B30] p-4 rounded-xl shadow-lg">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-gray-400">Compliance Rate</p>
              <p className="text-2xl font-bold text-white mt-1">{workerData.complianceRate || 0}%</p>
              <div className="mt-2">
                <span className={`text-xs font-medium ${
                  workerData.complianceRate < 90 ? 'text-yellow-400' : 'text-green-400'
                }`}>
                  {workerData.complianceRate < 90 ? 'Needs Improvement' : 'Good'}
                </span>
              </div>
            </div>
            <div className="p-4 bg-green-500/10 rounded-lg flex items-center justify-center ml-6">
              <ShieldCheck className="h-8 w-8 text-blue-400" />
            </div>
          </div>
        </div>

        {/* Last Violation Card */}
        <div className="bg-[#2A2B30] p-4 rounded-xl shadow-lg">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-gray-400">Last Violation</p>
              <p className="text-2xl font-bold text-white mt-1">
                {workerData.lastViolationDate 
                  ? format(parseISO(workerData.lastViolationDate), 'MMM d, yyyy') 
                  : 'None'}
              </p>
              <div className="mt-2">
                <span className="text-xs text-gray-400 font-medium">
                  {workerData.violationHistory?.[0]?.type || 'No violations'}
                </span>
              </div>
            </div>
            <div className="p-4 bg-yellow-500/10 rounded-lg flex items-center justify-center ml-6">
              <Calendar className="h-8 w-8 text-yellow-400" />
            </div>
          </div>
        </div>
      </div>

      {/* Violation History Section */}
      <div className="bg-[#2A2B30] p-6 shadow-lg rounded-xl mb-8">
        <table className="min-w-full">
            <thead>
              <tr className="grid grid-cols-4 gap-4 mb-4 text-xs md:text-sm font-semibold text-white">
                <th className="bg-[#19325C] px-4 py-2 rounded-lg text-left">Date & Time</th>
                <th className="bg-[#19325C] px-4 py-2 rounded-lg text-left">Violation Type</th>
                <th className="bg-[#19325C] px-4 py-2 rounded-lg text-left">Camera Location</th>
                <th className="bg-[#19325C] px-4 py-2 rounded-lg text-left">Action</th>
              </tr>
            </thead>
            <tbody className="space-y-2">
              {filterViolations().length > 0 ? (
                filterViolations().map((violation, index) => (
                  <tr 
                    key={index} 
                    className="grid grid-cols-4 gap-12 bg-[#2A2B30] rounded-lg shadow-sm border border-gray-700 p-4 hover:bg-[#3A3B40] transition-colors items-center"
                  >
                    <td className="text-gray-300">
                      {format(parseISO(violation.date), 'MMM d, yyyy hh:mm a')}
                    </td>
                    <td className="text-gray-300">{violation.type}</td>
                    <td className="text-gray-300">{violation.cameraLocation}</td>
                    <td className="text-gray-300">
                      <button className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-[#5388DF] rounded-md hover:bg-[#19325C] transition-colors">
                        <FaEye className="mr-2" />
                        View
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan="4" className="px-6 py-4 text-center text-sm text-gray-400">
                    No violation records found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
      </div>
    </div>
  );
}

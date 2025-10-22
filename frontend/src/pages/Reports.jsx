//frontend/src/pages/Reports.jsx
import React, { useState, useEffect } from "react";
import API from "../api"; // your axios instance
import { FiSearch, FiDownload, FiCheck } from "react-icons/fi";
import { HiOutlinePrinter } from "react-icons/hi";
import { FaMapMarkerAlt, FaUserAlt, FaChartLine } from 'react-icons/fa';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, Legend, AreaChart, Area } from 'recharts';

export default function Reports() {
const [selectedPeriod, setSelectedPeriod] = useState("Today");

  const [stats, setStats] = useState({
    total_incidents: 0,
    total_workers_involved: 0,
    violation_resolution_rate: 0,
    high_risk_locations: 0,
  });

  const [violationsData, setViolationsData] = useState([]);
  const [offendersData, setOffendersData] = useState([]);

  useEffect(() => {
  const fetchReports = async () => {
    try {

      const token = localStorage.getItem("token");
      const periodParam =
        selectedPeriod === "Last Week"
          ? "last_week"
          : selectedPeriod === "Last Month"
          ? "last_month"
          : "today";

      const response = await API.get(`/reports?period=${periodParam}`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      setCameraData(response.data.camera_data || []);
      setWorkerData(response.data.worker_data || []);

      setStats({
      total_incidents: response.data.total_incidents,
      total_workers_involved: response.data.total_workers_involved,
      violation_resolution_rate: response.data.violation_resolution_rate,
      high_risk_locations: response.data.high_risk_locations,
    });


      setViolationsData(response.data.most_violations);
      setOffendersData(
        response.data.top_offenders.map((o, i) => ({
          ...o,
          color: ["#34D399", "#38BDF8", "#F472B6", "#FDE68A", "#F59E0B"][i % 5],
        }))
      );
    } catch (err) {
      console.error("Error fetching reports:", err);
    }
  };

  fetchReports();
}, [selectedPeriod]);

 
const [cameraData, setCameraData] = useState([]);
const [workerData, setWorkerData] = useState([]);

const [performanceData, setPerformanceData] = useState([]);
const [avgResponseTime, setAvgResponseTime] = useState(0);

useEffect(() => {
  const fetchPerformance = async () => {
    try {
      const token = localStorage.getItem("token");
      const periodParam =
        selectedPeriod === "Last Week"
          ? "last_week"
          : selectedPeriod === "Last Month"
          ? "last_month"
          : "today";

      const res = await API.get(`/reports/performance?period=${periodParam}`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      setPerformanceData(res.data.performance_over_time || []);
      setAvgResponseTime(res.data.average_response_time || 0);
    } catch (err) {
      console.error("Error fetching performance data:", err);
    }
  };

  fetchPerformance();
}, [selectedPeriod]);

const handleExport = async () => {
  try {
    const token = localStorage.getItem("token");
    const periodParam =
      selectedPeriod === "Last Week"
        ? "last_week"
        : selectedPeriod === "Last Month"
        ? "last_month"
        : "today";

    const res = await API.get(`/reports/export?period=${periodParam}`, {
      headers: { Authorization: `Bearer ${token}` },
      responseType: "blob"
    });

    // Generate filename with current date on the frontend
    const now = new Date();
    const yyyy = now.getFullYear();
    const mm = String(now.getMonth() + 1).padStart(2, "0");
    const dd = String(now.getDate()).padStart(2, "0");
    const dateStr = `${yyyy}${mm}${dd}`; // e.g., 20251019

    const filename = `report_${periodParam}_${getDateRangeLabel().replace(/[^a-zA-Z0-9]/g, "_")}.csv`;


    // Create downloadable link
    const url = window.URL.createObjectURL(new Blob([res.data]));
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    link.remove();

  } catch (err) {
    console.error("Error exporting report:", err);
  }
};

const getDateRangeLabel = () => {
  const now = new Date();
  let startDate, endDate;

  if (selectedPeriod === "Today") {
    return now.toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
  }

  if (selectedPeriod === "Last Week") {
    endDate = new Date(now);
    startDate = new Date(now);
    startDate.setDate(now.getDate() - 7);
  } else if (selectedPeriod === "Last Month") {
    endDate = new Date(now);
    startDate = new Date(now);
    startDate.setMonth(now.getMonth() - 1);
  }

  const format = (date) =>
    date.toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });

  return `${format(startDate)} - ${format(endDate)}`;
};


const handlePrint = () => {
  const printContents = document.getElementById("printable-reports").innerHTML;
  const newWindow = window.open("", "_blank");
  newWindow.document.write(`
    <html>
      <head>
        <title>Reports - ${selectedPeriod}</title>
        <style>
          @media print {
            @page { size: A4; margin: 20mm; }
          }

          body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background-color: #fff;
            color: #111827;
            padding: 20px;
          }

          h1, h2, h3 {
            font-weight: 600;
            color: #111827;
          }

          h1 {
            text-align: center;
            font-size: 24px;
            margin-bottom: 10px;
          }

          h2 {
            border-bottom: 2px solid #5388DF;
            padding-bottom: 5px;
            margin-top: 30px;
            margin-bottom: 15px;
            color: #1E40AF;
          }

          .cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
          }

          .card {
            background-color: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
          }

          .card h3 {
            font-size: 14px;
            color: #6B7280;
            margin-bottom: 6px;
          }

          .card p {
            font-size: 22px;
            font-weight: bold;
            color: #2563EB;
          }

          table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            margin-bottom: 20px;
          }

          th, td {
            border: 1px solid #e5e7eb;
            padding: 8px 10px;
            text-align: left;
            font-size: 13px;
          }

          th {
            background-color: #f3f4f6;
            font-weight: 600;
            color: #374151;
          }

          tr:nth-child(even) {
            background-color: #f9fafb;
          }

          .chart-container {
            text-align: center;
            margin: 20px 0;
          }

          .chart-container img {
            max-width: 100%;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
          }

          footer {
            text-align: center;
            font-size: 12px;
            color: #6B7280;
            margin-top: 40px;
          }
        </style>
      </head>
      <body>
        <h1>Workplace Violation Report - ${selectedPeriod}</h1>
        <p style="text-align:center; font-size:14px; color:#6B7280; margin-top:0;">
          ${getDateRangeLabel()}
        </p>

        <div class="cards">
          <div class="card">
            <h3>Total Incidents</h3>
            <p>${stats.total_incidents}</p>
          </div>
          <div class="card">
            <h3>Total Workers Involved</h3>
            <p>${stats.total_workers_involved}</p>
          </div>
          <div class="card">
            <h3>Violation Resolution Rate</h3>
            <p style="color:#059669">${stats.violation_resolution_rate}%</p>
          </div>
          <div class="card">
            <h3>High Risk Locations</h3>
            <p style="color:#DC2626">${stats.high_risk_locations}</p>
          </div>
        </div>

        <h2>Camera Locations</h2>
        <table>
          <thead>
            <tr><th>Location</th><th>Violations</th><th>Risk Level</th></tr>
          </thead>
          <tbody>
            ${cameraData.map(loc => `
              <tr>
                <td>${loc.location}</td>
                <td>${loc.violations}</td>
                <td>${loc.risk}</td>
              </tr>`).join('')}
          </tbody>
        </table>

        <h2>Violation Resolution Rate (by Worker)</h2>
        <table>
          <thead>
            <tr><th>Rank</th><th>Worker</th><th>Violations</th><th>Resolved</th><th>Resolution Rate</th></tr>
          </thead>
          <tbody>
            ${workerData.map(w => `
              <tr>
                <td>${w.rank}</td>
                <td>${w.name}</td>
                <td>${w.violations}</td>
                <td>${w.resolved}</td>
                <td>${w.resolution_rate}%</td>
              </tr>`).join('')}
          </tbody>
        </table>

        <div class="chart-container">
          <h2>Charts & Graphs</h2>
          <p>(Refer to full dashboard for interactive charts)</p>
          <img src="data:image/png;base64,${window.html2canvas ? '' : ''}" alt="Charts Preview Placeholder">
        </div>

        <footer>
          Generated automatically by Safety Monitoring System • ${new Date().toLocaleString()}
        </footer>
      </body>
    </html>
  `);

  newWindow.document.close();
  newWindow.focus();
  newWindow.print();
};




  return (
    <div className="min-h-screen bg-[#1E1F23] text-gray-100 p-6" id="printable-reports">

    
      {/* Header */}
      <header className="bg-[#2A2B30] px-5 py-3 rounded-xl shadow-lg mb-8">
        <h1 className="text-2xl font-bold text-gray-100">REPORTS</h1>
      </header>

      {/* Search and Action Buttons */}
      <div className="flex flex-wrap items-center gap-4 mb-6">
        <div className="relative flex-1 min-w-[250px] max-w-md">
          <FiSearch className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search ..."
            className="w-full pl-12 pr-4 py-3 border border-gray-700 rounded-lg shadow-sm text-sm focus:outline-none focus:ring-2 focus:ring-[#5388DF] bg-[#2A2B30] text-gray-200"
          />
        </div>
        
       <button
          onClick={handleExport}
          className="flex items-center gap-2 px-5 py-3 text-sm font-medium text-white bg-[#5388DF] rounded-lg hover:bg-[#19325C] transition-colors"
        >
          <FiDownload className="w-4 h-4" />
          Export
        </button>

        
        <button
          onClick={handlePrint}
          className="flex items-center gap-2 px-5 py-3 text-sm font-medium text-white bg-[#5388DF] rounded-lg hover:bg-[#19325C] transition-colors"
        >
          <HiOutlinePrinter className="w-4 h-4" />
          Print
        </button>


        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => setSelectedPeriod('Last Month')}
            className={`px-4 py-2 text-sm rounded-lg transition-colors ${
              selectedPeriod === 'Last Month' 
                ? 'bg-[#5388DF] text-white font-medium' 
                : 'bg-[#2A2B30] text-gray-300 hover:bg-[#19325C]'
            }`}
          >
            Last Month
          </button>
          <button
            onClick={() => setSelectedPeriod('Last Week')}
            className={`px-4 py-2 text-sm rounded-lg transition-colors ${
              selectedPeriod === 'Last Week' 
                ? 'bg-[#5388DF] text-white font-medium' 
                : 'bg-[#2A2B30] text-gray-300 hover:bg-[#19325C]'
            }`}
          >
            Last Week
          </button>
          <button
            onClick={() => setSelectedPeriod('Today')}
            className={`flex items-center gap-2 px-4 py-2 text-sm rounded-lg transition-colors ${
              selectedPeriod === 'Today' 
                ? 'bg-[#5388DF] text-white font-medium' 
                : 'bg-[#2A2B30] text-gray-300 hover:bg-[#19325C]'
            }`}
          >
            {selectedPeriod === 'Today' && <FiCheck className="w-4 h-4" />}
            Today
          </button>
        </div>
      </div>
 {stats.total_incidents === 0 &&
 stats.total_workers_involved === 0 &&
 violationsData.length === 0 &&
 offendersData.length === 0 ? (
  <div className="text-center text-gray-400 text-sm my-6">
    No data for {selectedPeriod}.
  </div>
) : (
  <>
      {/* Statistics Cards */}
<div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
  <div className="bg-[#2A2B30] rounded-xl shadow-lg p-6 border border-gray-700">
    <h3 className="text-sm font-medium text-gray-400 mb-2">Total Incidents</h3>
    <p className="text-4xl font-bold text-[#5388DF]">{stats.total_incidents}</p>
  </div>

  <div className="bg-[#2A2B30] rounded-xl shadow-lg p-6 border border-gray-700">
    <h3 className="text-sm font-medium text-gray-400 mb-2">Total Workers Involved</h3>
    <p className="text-4xl font-bold text-[#5388DF]">{stats.total_workers_involved}</p>
  </div>

  <div className="bg-[#2A2B30] rounded-xl shadow-lg p-6 border border-gray-700">
    <h3 className="text-sm font-medium text-gray-400 mb-2">Violation Resolution Rate</h3>
    <p className="text-4xl font-bold text-green-500">{stats.violation_resolution_rate}%</p>
  </div>


  <div className="bg-[#2A2B30] rounded-xl shadow-lg p-6 border border-gray-700">
    <h3 className="text-sm font-medium text-gray-400 mb-2">High Risk Locations</h3>
    <p className="text-4xl font-bold text-red-500">{stats.high_risk_locations}</p>
  </div>
</div>


      {/* Charts Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Most Violations Chart */}
      <div className="bg-[#2A2B30] rounded-xl shadow-lg p-6 border border-gray-700">
        <h3 className="text-xl font-semibold text-gray-200 mb-4">Most Violations</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={violationsData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="name" 
              tick={{ fontSize: 12, fill: '#9CA3AF' }}
              stroke="#9CA3AF"
              axisLine={false}
            />
            <YAxis 
              tick={{ fontSize: 12, fill: '#9CA3AF' }}
              stroke="#9CA3AF"
              axisLine={false}
              ticks={[0, 2, 4, 6]}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1E1F23', 
                border: '1px solid #374151',
                borderRadius: '0.5rem',
                color: '#E5E7EB'
              }}
              cursor={false}  // This removes the hover effect
            />
            <Bar 
              dataKey="violators" 
              fill="#5388DF" 
              radius={[4, 4, 0, 0]}
              className="hover:opacity-90 transition-opacity"  // Subtle hover effect
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

        {/* Top Offenders Chart */}        
<div className="bg-[#2A2B30] rounded-xl shadow-lg p-6 border border-gray-700">
  <h3 className="text-xl font-semibold text-gray-200 mb-6">Top Offenders</h3>
  <ResponsiveContainer width="100%" height={offendersData.length * 70 + 30}>
    <BarChart data={offendersData} 
      layout="vertical"
      margin={{ top: 5, right: 30, left: 10, bottom: 5 }}
      barSize={40}
    >
      <CartesianGrid horizontal={false} stroke="#374151" strokeDasharray="3 3" />
      
      <XAxis 
        type="number"
        tick={{ fill: '#9CA3AF', fontSize: 12 }}
        axisLine={false}
        tickLine={false}
      />
      
      <YAxis 
        type="category" 
        dataKey="name"
        tick={{ fill: '#E5E7EB', fontSize: 12, fontWeight: 500 }}
        axisLine={false}
        tickLine={false}
        width={100}
      />
      
      <Tooltip 
        contentStyle={{ 
          backgroundColor: '#2A2B30', 
          border: '1px solid #4B5563',
          borderRadius: '0.5rem',
          color: '#E5E7EB',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.2)'
        }}
        cursor={false}
        formatter={(value) => [<span style={{ color: '#3B82F6' }}>{value} violations</span>]}
        labelFormatter={(name) => <span className="text-white font-medium">Worker: {name}</span>}
      />
      
      <Bar 
        dataKey="value" 
        radius={[0, 4, 4, 0]}
        label={{
          position: 'right',
          fill: '#E5E7EB',
          fontSize: 12,
          formatter: (value) => `${value} violations`,
          style: { textShadow: '0 0 4px rgba(0,0,0,0.7)' } // Adds text shadow for better visibility
        }}
        isAnimationActive={false} // Disables animation to prevent text flicker
      >
        {offendersData.map((entry, index) => (
          <Cell 
            key={`cell-${index}`} 
            fill={entry.color}
            stroke="#2A2B30"
            strokeWidth={2}
            className="hover:opacity-90 transition-opacity" // Subtle hover effect
          />
        ))}
      </Bar>
    </BarChart>
  </ResponsiveContainer>
</div>
</div>
     
    




      {/* Camera Location & Worker Rankings */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Location-Based Hotspots */}
        <div className="bg-[#2A2B30] rounded-xl shadow-lg p-6 border border-gray-700">
          <div className="flex items-center gap-2 mb-4">
            <FaMapMarkerAlt className="text-red-500" />
            <h3 className="text-xl font-semibold text-gray-200">Camera Location</h3>
          </div>
          <div className="max-h-[400px] overflow-y-auto pr-2 space-y-3 scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800">
            {cameraData.map((location, index) => (
              <div 
                key={index}
                className="bg-[#1E1F23] rounded-lg p-4 border border-gray-700 hover:bg-[#3A3B40] transition-colors"
              >
                <div className="flex justify-between items-center">
                  <div>
                    <p className="text-gray-200 font-medium">{location.location}</p>
                    <p className="text-gray-400 text-sm">{location.violations} violations</p>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                    location.risk === 'High' ? 'bg-red-500/20 text-red-400' :
                    location.risk === 'Medium' ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-green-500/20 text-green-400'
                  }`}>
                    {location.risk}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Worker Rankings */}
        <div className="bg-[#2A2B30] rounded-xl shadow-lg p-6 border border-gray-700">
          <div className="flex items-center gap-2 mb-4">
          <FaUserAlt className="text-[#5388DF]" />
            <h3 className="text-xl font-semibold text-gray-200">Violation Resolution Rate</h3>
          </div>
                  <div className="max-h-[400px] overflow-y-auto pr-2 space-y-3 scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800">
            {workerData.map((worker) => (
              <div 
                key={worker.rank}
                className="bg-[#1E1F23] rounded-lg p-4 border border-gray-700 hover:bg-[#3A3B40] transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="text-2xl font-bold text-gray-500 w-8">
                      {worker.rank}
                    </span>
                    <div>
                      <p className="text-gray-200 font-medium">{worker.name}</p>
                      <p className="text-gray-400 text-sm">{worker.violations} violations</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className={`text-lg font-bold ${
                      worker.resolution_rate >= 95 ? 'text-green-500' :
                      worker.resolution_rate >= 80 ? 'text-yellow-500' :
                      'text-red-500'
                    }`}>
                      {worker.resolution_rate}%
                    </p>
                    <p className="text-gray-400 text-xs">Resolution Rate</p>
                    <p className="text-gray-400 text-sm">
                      {worker.resolved} resolved / {worker.violations} total
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

       {/* Performance Over Time - Area Chart */}
       <div className="bg-[#2A2B30] rounded-xl shadow-lg p-6 border border-gray-700 mb-6">
        <div className="flex items-center gap-2 mb-4">
          <FaChartLine className="text-[#5388DF]" />
          <h3 className="text-xl font-semibold text-gray-200">Performance Over Time</h3>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={performanceData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="date" tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} />
            <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} />
            <Tooltip contentStyle={{ backgroundColor: '#1E1F23', border: '1px solid #374151', borderRadius: '0.5rem', color: '#E5E7EB' }} />
            <Legend wrapperStyle={{ paddingTop: '10px' }} />
            <Area type="monotone" dataKey="violations" stroke="#EF4444" fill="#EF4444" fillOpacity={0.6} />
            <Area type="monotone" dataKey="compliance" stroke="#10B981" fill="#10B981" fillOpacity={0.6} />
          </AreaChart>
        </ResponsiveContainer>

      </div>

      {/* Average Response Time Chart */}
      <div className="bg-[#2A2B30] rounded-xl shadow-lg p-6 border border-gray-700">
        <h3 className="text-xl font-semibold text-gray-200 mb-4">Average Response Time (min)</h3>
        <div className="text-4xl font-bold text-[#5388DF] mb-4">
          {avgResponseTime} min
        </div>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={performanceData.map(p => ({ date: p.date, time: p.violations } ))}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="date" tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} />
            <YAxis tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} />
            <Tooltip contentStyle={{ backgroundColor: '#1E1F23', border: '1px solid #374151', borderRadius: '0.5rem', color: '#E5E7EB' }} />
            <Line type="monotone" dataKey="time" stroke="#5388DF" strokeWidth={2} dot={{ r: 6 }} activeDot={{ r: 8 }} />
          </LineChart>
        </ResponsiveContainer>

              </div>
                </>
        )}
    </div>
    
  );
  
}
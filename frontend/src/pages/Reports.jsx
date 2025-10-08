import React, { useState } from 'react';
import { FiSearch, FiDownload, FiCheck } from 'react-icons/fi';
import { HiOutlinePrinter } from 'react-icons/hi';
import { FaMapMarkerAlt, FaUserAlt, FaChartLine } from 'react-icons/fa';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, Legend, AreaChart, Area } from 'recharts';

export default function Reports() {
  const [selectedPeriod, setSelectedPeriod] = useState('Today');

  // Sample data for Most Violations
  const violationsData = [
    { name: 'No Helmet', violators: 6 },
    { name: 'No Vest', violators: 3 },
    { name: 'No Gloves', violators: 2 },
    { name: 'No Boots', violators: 1 },
  ];

  // Sample data for Top Offenders
  const offendersData = [
    { name: 'Ayana Jade Alejo', value: 4, color: '#34D399' },
    { name: 'Naila Haliluddin', value: 3, color: '#38BDF8'  },
    { name: 'Athena Casino', value: 2, color: '#F472B6' },
    { name: 'Alfaith Luzon', value: 1, color: '#FDE68A' },
  ];

  // Sample data for Average Response Time
  const responseTimeData = [
    { date: 'Dec 14', time: 10 },
    { date: 'Dec 15', time: 20 },
    { date: 'Dec 16', time: 30 },
    { date: 'Dec 17', time: 20 },
    { date: 'Dec 18', time: 30 },
    { date: 'Dec 19', time: 10 },
    { date: 'Dec 20', time: 40 },
  ];

  // New data for Trends Over Time
  const trendsData = [
    { month: 'Jan', violations: 12, compliance: 88 },
    { month: 'Feb', violations: 8, compliance: 92 },
    { month: 'Mar', violations: 15, compliance: 85 },
    { month: 'Apr', violations: 10, compliance: 90 },
    { month: 'May', violations: 6, compliance: 94 },
    { month: 'Jun', violations: 9, compliance: 91 },
  ];

  // Location-based hotspots data
  const locationData = [
    { location: 'Camera 1 - Entrance', violations: 25, risk: 'High' },
    { location: 'Camera 2 - Equipment Area', violations: 18, risk: 'Medium' },
    { location: 'Camera 3 - Storage', violations: 12, risk: 'Medium' },
    { location: 'Camera 4 - Ground Floor', violations: 5, risk: 'Low' },
    { location: 'Camera 5 - 3rd Floor', violations: 5, risk: 'Low' },
    { location: 'Camera 6 - Rooftop', violations: 5, risk: 'Low' },
  ];

  // PPE Compliance data
    const complianceData = [
      { item: 'Helmet', compliant: 30, violation: 70 },        
      { item: 'Vest', compliant: 65, violation: 35 },   
      { item: 'Gloves', compliant: 85, violation: 15 },        
      { item: 'Safety Shoes', compliant: 95, violation: 5 },  
    ];

  // Worker Rankings
  const workerRankings = [
    { rank: 1, name: 'Maria Santos', violations: 0, compliance: 100 },
    { rank: 2, name: 'Pedro Garcia', violations: 1, compliance: 98 },
    { rank: 3, name: 'Ana Cruz', violations: 2, compliance: 95 },
    { rank: 4, name: 'Juan Dela Cruz', violations: 3, compliance: 92 },
    { rank: 5, name: 'Carlos Reyes', violations: 5, compliance: 88 },
  ];

  return (
    <div className="min-h-screen bg-[#1E1F23] text-gray-100 p-6">
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
        
        <button className="flex items-center gap-2 px-5 py-3 text-sm font-medium text-white bg-[#5388DF] rounded-lg hover:bg-[#19325C] transition-colors">
          <FiDownload className="w-4 h-4" />
          Export
        </button>
        
        <button className="flex items-center gap-2 px-5 py-3 text-sm font-medium text-white bg-[#5388DF] rounded-lg hover:bg-[#19325C] transition-colors">
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

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
        {/* Total Incidents Card */}
        <div className="bg-[#2A2B30] rounded-xl shadow-lg p-6 border border-gray-700">
          <h3 className="text-sm font-medium text-gray-400 mb-2">Total incidents</h3>
          <p className="text-4xl font-bold text-[#5388DF]">10</p>
        </div>

        {/* Total Workers Involved Card */}
        <div className="bg-[#2A2B30] rounded-xl shadow-lg p-6 border border-gray-700">
          <h3 className="text-sm font-medium text-gray-400 mb-2">Total workers involved</h3>
          <p className="text-4xl font-bold text-[#5388DF]">16</p>
        </div>

        {/* Overall Compliance Rate */}
        <div className="bg-[#2A2B30] rounded-xl shadow-lg p-6 border border-gray-700">
          <h3 className="text-sm font-medium text-gray-400 mb-2">Overall Compliance</h3>
          <p className="text-4xl font-bold text-green-500">91%</p>
        </div>

        {/* High Risk Locations */}
        <div className="bg-[#2A2B30] rounded-xl shadow-lg p-6 border border-gray-700">
          <h3 className="text-sm font-medium text-gray-400 mb-2">High Risk Locations</h3>
          <p className="text-4xl font-bold text-red-500">2</p>
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
    <BarChart 
      data={[...offendersData].sort((a, b) => b.value - a.value)} 
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
            {locationData.map((location, index) => (
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
            <h3 className="text-xl font-semibold text-gray-200">Worker Compliance Score</h3>
          </div>
                  <div className="max-h-[400px] overflow-y-auto pr-2 space-y-3 scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800">
            {workerRankings.map((worker) => (
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
                      worker.compliance >= 95 ? 'text-green-500' :
                      worker.compliance >= 90 ? 'text-yellow-500' :
                      'text-red-500'
                    }`}>
                      {worker.compliance}%
                    </p>
                    <p className="text-gray-400 text-xs">Compliance</p>
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
          <AreaChart data={trendsData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="month" 
              tick={{ fontSize: 12, fill: '#9CA3AF' }}
              stroke="#9CA3AF"
              axisLine={false}
            />
            <YAxis 
              tick={{ fontSize: 12, fill: '#9CA3AF' }}
              stroke="#9CA3AF"
              axisLine={false}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1E1F23', 
                border: '1px solid #374151',
                borderRadius: '0.5rem',
                color: '#E5E7EB'
              }}
            />
            <Legend 
              wrapperStyle={{ paddingTop: '10px' }}
              formatter={(value) => (
                <span style={{ color: '#9CA3AF', fontSize: '12px' }}>
                  {value === 'violations' ? 'Violations' : 'Compliance %'}
                </span>
              )}
            />
            <Area 
              type="monotone" 
              dataKey="violations" 
              stroke="#EF4444" 
              fill="#EF4444" 
              fillOpacity={0.6}
            />
            <Area 
              type="monotone" 
              dataKey="compliance" 
              stroke="#10B981" 
              fill="#10B981" 
              fillOpacity={0.6}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Average Response Time Chart */}
      <div className="bg-[#2A2B30] rounded-xl shadow-lg p-6 border border-gray-700">
        <h3 className="text-xl font-semibold text-gray-200 mb-4">Average Response Time (min)</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={responseTimeData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="date" 
              tick={{ fontSize: 12, fill: '#9CA3AF' }}
              stroke="#9CA3AF"
              axisLine={false}
            />
            <YAxis 
              tick={{ fontSize: 12, fill: '#9CA3AF' }}
              stroke="#9CA3AF"
              axisLine={false}
              ticks={[0, 10, 20, 30, 40]}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1E1F23', 
                border: '1px solid #374151',
                borderRadius: '0.5rem',
                color: '#E5E7EB'
              }}
            />
            <Line 
              type="monotone" 
              dataKey="time" 
              stroke="#5388DF" 
              strokeWidth={2}
              dot={{ fill: '#5388DF', r: 6 }}
              activeDot={{ r: 8 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
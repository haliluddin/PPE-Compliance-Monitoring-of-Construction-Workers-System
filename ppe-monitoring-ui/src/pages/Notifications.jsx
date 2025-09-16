import { useState } from "react";
import DateRangePicker from "../components/DateRangePicker";
import CustomTable from "../components/CustomTable";

export default function Notifications() {
  const [dateRange, setDateRange] = useState({ startDate: null, endDate: null });
  const [selectedCamera, setSelectedCamera] = useState("");
  const [selectedViolation, setSelectedViolation] = useState("");

  const handleDateChange = (range) => {
    setDateRange(range);
    console.log("Selected range:", range);
  };

  const columns = [
    { key: "id", label: "ID" },
    { key: "name", label: "Name" },
    { key: "role", label: "Role" },
  ];

  const data = [
    { id: 1, name: "John Doe", role: "Camera Operator" },
    { id: 2, name: "Jane Smith", role: "Supervisor" },
  ];

  const actions = [
    {
      label: "Edit",
      color: "bg-blue-500 hover:bg-blue-600",
      onClick: (row) => alert(`Edit ${row.name}`),
    },
    {
      label: "Delete",
      color: "bg-red-500 hover:bg-red-600",
      onClick: (row) => alert(`Delete ${row.name}`),
    },
  ];

  return (
    <div className="p-6 text-[#19325C] space-y-6">
      {/* Page Title */}
      <h1 className="text-2xl font-bold">Notifications</h1>

      {/* Filters */}
      <div className="flex flex-wrap justify-between items-end gap-4">
        {/* Camera Filter */}
        <div className="flex flex-col gap-1">
          <label className="text-sm font-bold">Camera</label>
          <select
            value={selectedCamera}
            onChange={(e) => setSelectedCamera(e.target.value)}
            className="px-4 py-1 border border-gray-300 rounded-lg bg-white text-[#19325C] focus:outline-none focus:ring-2 focus:ring-[#21005D]"
          >
            <option value="">Select an option</option>
            <option value="Camera 1">Camera 1</option>
            <option value="Camera 2">Camera 2</option>
          </select>
        </div>

        {/* Violation Filter */}
        <div className="flex flex-col gap-1">
          <label className="text-sm font-bold">Violation</label>
          <select
            value={selectedViolation}
            onChange={(e) => setSelectedViolation(e.target.value)}
            className="px-4 py-1 border border-gray-300 rounded-lg bg-white text-[#19325C] focus:outline-none focus:ring-2 focus:ring-[#21005D]"
          >
            <option value="">Select an option</option>
            <option value="Violation 1">Violation 1</option>
            <option value="Violation 2">Violation 2</option>
          </select>
        </div>

        {/* Date Picker */}
        {/* <div className="flex flex-col gap-1">
          <label className="text-sm font-bold">Date Range</label>
          <DateRangePicker onChange={handleDateChange} />
        </div> */}
      </div>

      {/* Table */}
      <CustomTable columns={columns} data={data} actions={actions} />
    </div>
  );
}

import { useState, useEffect } from "react";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";

export default function DateRangePicker({ onChange }) {
  const [startDate, setStartDate] = useState(null);
  const [endDate, setEndDate] = useState(null);

  // Notify parent when dates change
  useEffect(() => {
    if (onChange) {
      onChange({ startDate, endDate });
    }
  }, [startDate, endDate, onChange]);

  return (
    <div className="flex items-center gap-4">
      {/* From Date */}
      <div className="flex flex-col">
        <label className="text-sm text-[#19325C] mb-1">From</label>
        <DatePicker
          selected={startDate}
          onChange={(date) => setStartDate(date)}
          selectsStart
          startDate={startDate}
          endDate={endDate}
          placeholderText="Select start date"
          className="px-3 py-1 border border-[#21005D] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#21005D]"
        />
      </div>

      {/* To Date */}
      <div className="flex flex-col">
        <label className="text-sm text-[#19325C] mb-1">To</label>
        <DatePicker
          selected={endDate}
          onChange={(date) => setEndDate(date)}
          selectsEnd
          startDate={startDate}
          endDate={endDate}
          minDate={startDate}
          placeholderText="Select end date"
          className="px-3 py-2 border border-[#21005D] rounded-lg focus:outline-none focus:ring-2 focus:ring-[#21005D]"
        />
      </div>
    </div>
  );
}

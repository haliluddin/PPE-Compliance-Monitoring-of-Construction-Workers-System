import { FiUpload, FiCamera } from "react-icons/fi";
import ImageCard from "../components/ImageCard";
import { useState, useEffect } from "react";

export default function Camera() {
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const formattedDate = currentTime.toLocaleDateString();
  const formattedTime = currentTime.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <div className="p-8 text-[#19325C] bg-gray-50 min-h-screen">
      {/* ---------- Page Title ---------- */}
      <header className="mb-8">
        <h1 className="text-3xl font-bold mb-1">Camera Monitoring</h1>
        <p className="text-gray-600 max-w-2xl">
          View live camera feeds, upload recorded videos, and manage camera
          sources for real-time safety monitoring.
        </p>
      </header>

      {/* ---------- Filters & Actions ---------- */}
      <section className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 mb-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
          {/* Filter dropdown */}
          <div>
            <label className="block text-sm font-medium mb-1">
              Select Camera Group
            </label>
            <select className="w-full px-4 py-2 border border-gray-300 rounded-lg bg-white text-[#19325C] shadow-sm focus:outline-none focus:ring-2 focus:ring-[#19325C]">
              <option value="">Choose a group</option>
              <option>Entrance Area</option>
              <option>Main Site</option>
              <option>Warehouse</option>
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Filter cameras by location or category.
            </p>
          </div>

          {/* Action buttons */}
          <div className="flex gap-4 md:justify-end">
            <button className="flex items-center gap-2 px-4 py-2 bg-[#5388DF] text-white rounded-lg hover:bg-[#19325C] transition">
              <FiUpload size={18} />
              Upload Videos
            </button>

            <button className="flex items-center gap-2 px-4 py-2 bg-[#19325C] text-white rounded-lg hover:bg-[#5388DF] transition">
              <FiCamera size={18} />
              Add Camera
            </button>
          </div>
        </div>
      </section>

      {/* ---------- Status & Time Info ---------- */}
      <section className="flex flex-wrap items-center gap-4 mb-8">
        <div className="bg-[#19325C]/10 text-[#19325C] px-4 py-2 rounded-full text-sm font-medium">
          {formattedDate} • {formattedTime}
        </div>

        <div className="flex items-center gap-2 bg-green-100 text-green-700 px-4 py-2 rounded-full text-sm font-medium">
          <span className="w-3 h-3 rounded-full bg-green-500"></span>
          Compliance
        </div>

        <div className="flex items-center gap-2 bg-red-100 text-red-700 px-4 py-2 rounded-full text-sm font-medium">
          <span className="w-3 h-3 rounded-full bg-red-500"></span>
          Non-Compliance
        </div>
      </section>

      {/* ---------- Camera List ---------- */}
      <section>
        <h2 className="text-xl font-semibold mb-4">Available Cameras</h2>
        <p className="text-gray-600 text-sm mb-6">
          Click a camera to view live footage or recent recordings.
        </p>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          <ImageCard
            image="https://images.unsplash.com/photo-1535379453347-1ffd615e2e08?auto=format&fit=crop&w=800&q=80"
            title="Camera 1"
            time="12:45 PM"
          />
        </div>
      </section>
    </div>
  );
}

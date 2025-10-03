import { FiUpload, FiCamera, FiSearch, FiMaximize2, FiSettings, FiVideo, FiWifi, FiAlertTriangle } from "react-icons/fi";
import ImageCard from "../components/ImageCard";
import { useState, useEffect } from "react";


export default function Camera() {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [searchQuery, setSearchQuery] = useState("");
  const [cameras, setCameras] = useState([
    {
      image: "https://images.unsplash.com/photo-1535379453347-1ffd615e2e08?auto=format&fit=crop&w=800&q=80",
      title: "Camera 1",
      status: "LIVE",
    },
    {
      image: "https://images.unsplash.com/photo-1535379453347-1ffd615e2e08?auto=format&fit=crop&w=800&q=80",
      title: "Camera 2",
      status: "OFFLINE",
    },
    {
      image: "https://images.unsplash.com/photo-1535379453347-1ffd615e2e08?auto=format&fit=crop&w=800&q=80",
      title: "Camera 3",
      status: "NO SIGNAL",
    },
  ]);

  const [backendMsg, setBackendMsg] = useState(""); 

  // Update time every second
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Fetch backend status on mount
    useEffect(() => {
    const fetchBackendStatus = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000"); // FastAPI URL
        const data = await res.json();
        const statusText = data.database
          ? `${data.message} | DB: ${data.database}`
          : data.message;
        setBackendMsg(statusText);
      } catch (err) {
        console.error("Backend connection failed:", err);
        setBackendMsg("Cannot connect to backend ");
      }
    };
    fetchBackendStatus();
  }, []);


  const formattedDate = currentTime.toLocaleDateString();
  const formattedTime = currentTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  const stats = [
    { title: "TOTAL CAMERAS", value: "24", trend: "up", icon: <FiVideo className="text-[#5388DF]" size={32} /> },
    { title: "ACTIVE FEEDS", value: "23", trend: "up", icon: <FiWifi className="text-green-400" size={32} /> },
    { title: "NON-COMPLIANCE ALERTS", value: "1", trend: "warning", icon: <FiAlertTriangle className="text-amber-400" size={32} /> },
  ];

  const handleCameraClick = (title) => console.log(`Camera clicked: ${title}`);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const newCamera = {
        image: URL.createObjectURL(file),
        title: `Camera ${cameras.length + 1}`,
        status: "LIVE",
      };
      setCameras((prev) => [...prev, newCamera]);
    }
  };

  const handleRemoveCamera = (cameraTitle) => {
    setCameras((prev) => prev.filter((camera) => camera.title !== cameraTitle));
  };

  return (
    <div className="p-8 text-gray-100 bg-[#1E1F23] min-h-screen">
      {/* --- Backend Status --- */}
      <div className="mb-4 p-3 bg-[#2A2B30] rounded-lg text-white">
        <strong>Backend Status:</strong> {backendMsg || "Loading..."}
      </div>

      {/* System Stat Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {stats.map((stat, index) => (
          <div key={index} className="bg-[#2A2B30] px-5 py-3 rounded-xl shadow-lg">
            <h3 className="text-gray-400 text-sm mb-4">{stat.title}</h3>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-3xl font-bold text-white">{stat.value}</span>
              </div>
              {stat.icon}
            </div>
          </div>
        ))}
      </div>

      {/* Search & Filter Section */}
      <div className="flex flex-col md:flex-row gap-4 mb-8">
        <div className="relative flex-1">
          <input
            type="text"
            placeholder="Search Cameras"
            className="w-full bg-[#2A2B30] text-gray-200 pl-12 pr-4 py-3 rounded-lg border border-gray-700 focus:outline-none focus:border-[#5388DF]"
            onChange={(e) => setSearchQuery(e.target.value)}
            value={searchQuery}
          />
          <FiSearch className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
        </div>

        <div className="w-full md:w-64">
          <select className="w-full px-4 py-3 border border-gray-700 rounded-lg bg-[#2A2B30] text-gray-200 shadow-sm focus:outline-none focus:ring-2 focus:ring-[#5388DF]">
            <option value="">All Camera Groups</option>
            <option>Entrance Area</option>
            <option>Main Site</option>
            <option>Warehouse</option>
          </select>
        </div>

        <div className="flex gap-2">
          <input type="file" accept="image/*" className="hidden" id="image-upload" onChange={handleImageUpload} />
          <button
            className="px-4 py-3 bg-[#5388DF] text-white rounded-lg hover:bg-[#19325C] transition flex items-center gap-2"
            onClick={() => document.getElementById("image-upload").click()}
          >
            <FiUpload size={18} />
            <span className="hidden md:inline">Upload</span>
          </button>
          <button className="px-4 py-3 bg-[#19325C] text-white rounded-lg hover:bg-[#5388DF] transition flex items-center gap-2">
            <FiCamera size={18} />
            <span className="hidden md:inline">Add Camera</span>
          </button>
        </div>
      </div>

      {/* Status & Time */}
      <section className="flex flex-wrap items-center gap-4 mb-8">
        <div className="bg-[#2A2B30] text-gray-200 px-4 py-2 rounded-lg text-md font-medium">
          {formattedDate} | {formattedTime}
        </div>
        <div className="flex items-center gap-2 bg-green-100 text-green-700 px-4 py-2 rounded-lg text-sm font-medium">
          <span className="w-3 h-3 rounded-full bg-green-500"></span>
          Compliance
        </div>
        <div className="flex items-center gap-2 bg-red-100 text-red-700 px-4 py-2 rounded-lg text-sm font-medium">
          <span className="w-3 h-3 rounded-full bg-red-500"></span>
          Non-Compliance
        </div>
      </section>

      {/* Camera List */}
      <section>
        <h2 className="text-xl font-semibold mb-4 text-white">Available Cameras</h2>
        <p className="text-gray-300 text-sm mb-6">
          Click a camera to view live footage or recent recordings.
        </p>

        <div className="grid gap-6 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
          {cameras.map((camera, index) => (
            <ImageCard
              key={index}
              image={camera.image}
              title={camera.title}
              time={formattedTime}
              status={camera.status}
              onClick={() => handleCameraClick(camera.title)}
              onRemove={() => handleRemoveCamera(camera.title)}
              actionIcons={[
                { icon: <FiMaximize2 size={16} />, onClick: () => console.log(`Maximize ${camera.title}`) },
                { icon: <FiSettings size={16} />, onClick: () => console.log(`Settings ${camera.title}`) },
              ]}
            />
          ))}
        </div>
      </section>
    </div>
  );
}

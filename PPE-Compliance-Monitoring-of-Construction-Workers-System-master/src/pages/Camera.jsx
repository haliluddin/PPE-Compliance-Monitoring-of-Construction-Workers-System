import { FiUpload, FiCamera, FiSearch, FiMaximize2, FiSettings, FiVideo, FiWifi, FiAlertTriangle } from "react-icons/fi";
import ImageCard from "../components/ImageCard";
import { useState, useEffect, useRef } from "react";

const API_BASE = (typeof window !== "undefined" && window.__ENV && window.__ENV.API_BASE)
  || (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_API_BASE)
  || "http://127.0.0.1:9000";

const WS_BASE = (typeof window !== "undefined" && window.__ENV && window.__ENV.VITE_WS_URL)
  || (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_WS_URL)
  || API_BASE.replace(/^http/, "ws");

export default function Camera() {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [searchQuery, setSearchQuery] = useState("");
  const [cameras, setCameras] = useState([]);
  const [backendMsg, setBackendMsg] = useState("");
  const [selectedCamera, setSelectedCamera] = useState(null);
  const wsRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const fetchBackendStatus = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        setBackendMsg(data.triton_ready ? "Backend ready ✅" : "Backend not ready ⚠️");
      } catch (err) {
        console.error("Backend connection failed:", err);
        setBackendMsg("Cannot connect to backend ❌");
      }
    };
    fetchBackendStatus();
  }, []);

  useEffect(() => {
    if (wsRef.current) return;
    try {
      const ws = new WebSocket(`${WS_BASE}/ws`);
      wsRef.current = ws;
      ws.onopen = () => {
      };
      ws.onmessage = (ev) => {
        try {
          const payload = JSON.parse(ev.data);
          const meta = payload.meta || {};
          const jobId = meta.job_id ?? (meta.jobId ?? null);
          const annotated = payload.annotated_jpeg_b64 ?? payload.annotated_jpeg ?? null;
          if (!jobId) return;
          setCameras((prev) =>
            prev.map((cam) => {
              if (String(cam.job_id) !== String(jobId)) return cam;
              const next = { ...cam };
              if (annotated && (next.is_stream || next.status === "LIVE" || next.meta?.is_stream)) {
                next.frameSrc = `data:image/jpeg;base64,${annotated}`;
                next.status = "LIVE";
              } else {
                if (annotated) {
                  next.latestAnnotatedThumb = `data:image/jpeg;base64,${annotated}`;
                }
              }
              if (payload.people) {
                next.latest_people = payload.people;
              }
              return next;
            })
          );
        } catch (e) {
        }
      };
      ws.onclose = () => {
        wsRef.current = null;
      };
      ws.onerror = () => {
      };
    } catch (e) {
      wsRef.current = null;
    }
    return () => {
      if (wsRef.current) {
        try {
          wsRef.current.close();
        } catch {}
        wsRef.current = null;
      }
    };
  }, []);

  const formattedDate = currentTime.toLocaleDateString();
  const formattedTime = currentTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  const stats = [
    { title: "TOTAL CAMERAS", value: String(cameras.length), trend: "up", icon: <FiVideo className="text-[#5388DF]" size={32} /> },
    { title: "ACTIVE FEEDS", value: String(cameras.filter(c=>c.status==="LIVE").length), trend: "up", icon: <FiWifi className="text-green-400" size={32} /> },
    { title: "NON-COMPLIANCE ALERTS", value: String(cameras.reduce((acc,c)=>acc + (c.latest_people?.reduce((a,p)=>a + (p.violations?.length?1:0),0) || 0),0)), trend: "warning", icon: <FiAlertTriangle className="text-amber-400" size={32} /> },
  ];

  const handleCameraClick = (camera) => {
    setSelectedCamera(camera);
  };

  const handleRemoveCamera = (cameraJobId) => {
    setCameras((prev) => prev.filter((c) => String(c.job_id) !== String(cameraJobId)));
    if (selectedCamera && String(selectedCamera.job_id) === String(cameraJobId)) setSelectedCamera(null);
  };

  const createJob = async (file, opts = {}) => {
    const payload = {
      job_type: "video",
      camera_id: opts.camera_id ?? null,
      meta: {
        title: opts.title ?? file?.name ?? `upload-${Date.now()}`,
        source: opts.source ?? "upload",
        is_stream: !!opts.is_stream
      }
    };
    const res = await fetch(`${API_BASE}/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Failed to create job");
    const data = await res.json();
    return data.job_id;
  };

  const uploadJobVideo = async (jobId, file) => {
    const fd = new FormData();
    fd.append("file", file, file.name);
    const res = await fetch(`${API_BASE}/jobs/${jobId}/upload`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) {
      const txt = await res.text().catch(()=>"");
      throw new Error(`Upload failed: ${res.status} ${txt}`);
    }
    return await res.json();
  };

  const handleVideoUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const localPreview = URL.createObjectURL(file);
    const tempCam = {
      job_id: `tmp-${Date.now()}`,
      title: `Camera ${cameras.length + 1}`,
      status: "UPLOADING",
      videoUrl: localPreview,
      frameSrc: null,
      latest_people: [],
      is_stream: false,
      meta: { is_stream: false }
    };
    setCameras(prev => [...prev, tempCam]);
    let jobId;
    try {
      jobId = await createJob(file, { title: `Upload ${file.name}`, source: "camera-ui", is_stream: false });
      setCameras(prev => prev.map(c => c.job_id === tempCam.job_id ? ({ ...c, job_id: jobId, status: "UPLOADING", meta: { is_stream: false } }) : c));
    } catch (e) {
      setCameras(prev => prev.filter(c => c.job_id !== tempCam.job_id));
      console.error("create job error", e);
      return;
    }
    try {
      await uploadJobVideo(jobId, file);
      setCameras(prev => prev.map(c => (String(c.job_id) === String(jobId) ? ({ ...c, status: "PROCESSING", videoUrl: localPreview }) : c)));
    } catch (e) {
      setCameras(prev => prev.map(c => (String(c.job_id) === String(jobId) ? ({ ...c, status: "UPLOAD_FAILED" }) : c)));
      console.error("upload error", e);
    }
  };

  return (
    <div className="p-8 text-gray-100 bg-[#1E1F23] min-h-screen">
      <div className="mb-4 p-3 bg-[#2A2B30] rounded-lg text-white">
        <strong>Backend Status:</strong> {backendMsg || "Loading..."}
      </div>

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
          <input
            type="file"
            accept="video/*"
            className="hidden"
            id="video-upload"
            ref={fileInputRef}
            onChange={handleVideoUpload}
          />
          <button
            className="px-4 py-3 bg-[#5388DF] text-white rounded-lg hover:bg-[#19325C] transition flex items-center gap-2"
            onClick={() => fileInputRef.current && fileInputRef.current.click()}
          >
            <FiUpload size={18} />
            <span className="hidden md:inline">Upload Video</span>
          </button>
          <button className="px-4 py-3 bg-[#19325C] text-white rounded-lg hover:bg-[#5388DF] transition flex items-center gap-2">
            <FiCamera size={18} />
            <span className="hidden md:inline">Add Camera</span>
          </button>
        </div>
      </div>

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

      <section>
        <h2 className="text-xl font-semibold mb-4 text-white">Available Cameras</h2>
        <p className="text-gray-300 text-sm mb-6">Click a camera to view the live annotated frames or recent recordings.</p>

        <div className="grid gap-6 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
          {cameras
            .filter(c => c.title.toLowerCase().includes(searchQuery.toLowerCase()))
            .map((camera, index) => (
              <ImageCard
                key={camera.job_id || index}
                image={camera.frameSrc || camera.videoUrl}
                title={camera.title}
                time={formattedTime}
                status={camera.status}
                onClick={() => handleCameraClick(camera)}
                onRemove={() => handleRemoveCamera(camera.job_id)}
                actionIcons={[
                  { icon: <FiMaximize2 size={16} />, onClick: () => handleCameraClick(camera) },
                  { icon: <FiSettings size={16} />, onClick: () => console.log(`Settings ${camera.title}`) },
                ]}
              />
            ))}
        </div>
      </section>

      {selectedCamera && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-6">
          <div className="bg-[#111214] rounded-lg w-full max-w-4xl p-4">
            <div className="flex justify-between items-center mb-3">
              <h3 className="text-white font-semibold">{selectedCamera.title}</h3>
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-300">{selectedCamera.status}</span>
                <button className="text-gray-300 px-3 py-1 rounded bg-gray-800" onClick={() => setSelectedCamera(null)}>Close</button>
              </div>
            </div>
            <div className="w-full h-[60vh] bg-black rounded overflow-hidden flex items-center justify-center">
              {selectedCamera.frameSrc ? (
                <img alt="annotated" src={selectedCamera.frameSrc} className="object-contain w-full h-full" />
              ) : (
                <video controls src={selectedCamera.videoUrl} className="object-contain w-full h-full" />
              )}
            </div>
            <div className="mt-3 text-xs text-gray-300">
              <pre className="text-xs text-gray-400 max-h-32 overflow-auto">{JSON.stringify(selectedCamera.latest_people || [], null, 2)}</pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

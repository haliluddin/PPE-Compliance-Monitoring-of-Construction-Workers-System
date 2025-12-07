//frontend/src/pages/Camera.jsx
import { FiUpload, FiCamera, FiSearch, FiMaximize2, FiVideo, FiWifi, FiAlertTriangle } from "react-icons/fi";
import ImageCard from "../components/ImageCard";
import { useState, useEffect, useRef } from "react";
import { API_BASE, WS_BASE } from "../config";

export default function Camera() {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [searchQuery, setSearchQuery] = useState("");
  const [cameras, setCameras] = useState([]);
  const [backendMsg, setBackendMsg] = useState("");
  const wsRef = useRef(null);
  const fileInputRef = useRef(null);
  const wsPath = (WS_BASE || "").replace(/\/+$/, "") + "/ws";
  const [showAddModal, setShowAddModal] = useState(false);
  const [newCamName, setNewCamName] = useState("");
  const [newCamLocation, setNewCamLocation] = useState("");
  const [newCamRtsp, setNewCamRtsp] = useState("");
  const [addingCamera, setAddingCamera] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [selectedCameraJobId, setSelectedCameraJobId] = useState(null);

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
        setBackendMsg("Cannot connect to backend ❌");
      }
    };
    fetchBackendStatus();
  }, []);

  useEffect(() => {
    let mounted = true;
    async function loadCameras() {
      try {
        const res = await fetch(`${API_BASE}/cameras`);
        if (!res.ok) return;
        const data = await res.json();
        if (!mounted) return;
        const list = data.cameras || data || [];
        const normalized = list.map((c, i) => ({
          job_id: c.job_id ?? null,
          camera_id: c.id ?? c.camera_id ?? c.cameraId ?? null,
          title: c.location || c.name || `Camera ${c.id ?? i + 1}`,
          location: c.location || "",
          status: "IDLE",
          videoUrl: null,
          frameSrc: null,
          latest_people: [],
          meta: {}
        }));
        setCameras(normalized);
      } catch (e) {}
    }
    loadCameras();
    return () => { mounted = false; };
  }, []);

  useEffect(() => {
    if (wsRef.current) return;
    if (!WS_BASE) return;
    try {
      const ws = new WebSocket(wsPath);
      wsRef.current = ws;
      ws.onopen = () => {
        console.log("[WS] connected to", wsPath);
      };
      ws.onmessage = (ev) => {
        console.log("[WS] message raw:", ev.data);
        try {
          const payload = JSON.parse(ev.data);
          console.log("[WS] parsed payload:", payload);
          const meta = payload.meta || {};
          const jobId = meta.job_id ?? (meta.jobId ?? null);
          const annotated = payload.annotated_jpeg_b64 ?? payload.annotated_jpeg ?? null;
          if (!jobId) return;
          setCameras((prev) => {
            let found = false;
            const mapped = prev.map((cam) => {
              if (String(cam.job_id) !== String(jobId)) return cam;
              found = true;
              const nextCam = { ...cam };
              if (annotated) {
                nextCam.frameSrc = `data:image/jpeg;base64,${annotated}`;
                nextCam.latestAnnotatedThumb = `data:image/jpeg;base64,${annotated}`;
                nextCam.status = "LIVE";
              }
              if (payload.people) {
                nextCam.latest_people = payload.people;
              }
              nextCam.meta = { ...(nextCam.meta || {}), ...(meta || {}) };
              return nextCam;
            });
            if (!found) {
              const newCam = {
                job_id: jobId,
                title: `Camera ${jobId}`,
                status: annotated ? "LIVE" : "PROCESSING",
                videoUrl: null,
                frameSrc: annotated ? `data:image/jpeg;base64,${annotated}` : null,
                latestAnnotatedThumb: annotated ? `data:image/jpeg;base64,${annotated}` : null,
                latest_people: payload.people || [],
                meta: meta || {}
              };
              return [...mapped, newCam];
            }
            return mapped;
          });
        } catch (e) {
          console.error("[WS] parse error", e);
        }
      };
      ws.onclose = (ev) => {
        console.warn("[WS] closed", ev);
        wsRef.current = null;
      };
      ws.onerror = (err) => {
        console.error("[WS] error", err);
        wsRef.current = null;
      };
    } catch (e) {
      wsRef.current = null;
    }
    return () => {
      if (wsRef.current) {
        try { wsRef.current.close(); } catch {}
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
    if (!camera) return;
    setSelectedCameraJobId(camera.job_id ?? null);
  };

  const handleRemoveCamera = (cameraJobId) => {
    setCameras((prev) => {
      const removed = prev.find(c => String(c.job_id) === String(cameraJobId));
      if (removed && removed.videoUrl && removed.videoUrl.startsWith && removed.videoUrl.startsWith("blob:")) {
        try { URL.revokeObjectURL(removed.videoUrl); } catch (e) { /* ignore */ }
      }
      return prev.filter((c) => String(c.job_id) !== String(cameraJobId));
    });
    if (selectedCameraJobId && String(selectedCameraJobId) === String(cameraJobId)) setSelectedCameraJobId(null);
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
      title: file.name,
      status: "UPLOADING",
      videoUrl: localPreview,
      frameSrc: null,
      latest_people: [],
      is_stream: false,
      meta: { is_stream: false, title: file.name }
    };
    setCameras(prev => [...prev, tempCam]);
    let jobId;
    try {
      jobId = await createJob(file, { title: `Upload ${file.name}`, source: "camera-ui", is_stream: false });
      setCameras(prev => prev.map(c => c.job_id === tempCam.job_id ? ({ ...c, job_id: jobId, status: "UPLOADING", meta: { ...c.meta, job_id: jobId } }) : c));
    } catch (e) {
      setCameras(prev => prev.filter(c => c.job_id !== tempCam.job_id));
      return;
    }
    try {
      await uploadJobVideo(jobId, file);
      setCameras(prev => prev.map(c => (String(c.job_id) === String(jobId) ? ({ ...c, status: "PROCESSING", videoUrl: localPreview }) : c)));
    } catch (e) {
      setCameras(prev => prev.map(c => (String(c.job_id) === String(jobId) ? ({ ...c, status: "UPLOAD_FAILED" }) : c)));
    }
  };

  const createCameraOnServer = async ({ name, location, stream_url }) => {
    const payload = { name, location, stream_url };
    const res = await fetch(`${API_BASE}/cameras`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const txt = await res.text().catch(() => "");
      throw new Error(`Failed creating camera: ${res.status} ${txt}`);
    }
    return await res.json();
  };

  const startStreamOnServer = async ({ stream_url, camera_id }) => {
    const res = await fetch(`${API_BASE}/streams`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stream_url, camera_id })
    });
    if (!res.ok) {
      const txt = await res.text().catch(() => "");
      throw new Error(`Failed to start stream: ${res.status} ${txt}`);
    }
    return await res.json();
  };

  const handleAddCameraSubmit = async (e) => {
    e.preventDefault();
    setErrorMsg("");
    setAddingCamera(true);
    try {
      const camRes = await createCameraOnServer({ name: newCamName || `Camera`, location: newCamLocation || "", stream_url: newCamRtsp });
      const cameraId = camRes.camera_id ?? camRes.id;
      const streamRes = await startStreamOnServer({ stream_url: newCamRtsp, camera_id: cameraId });
      const jobId = streamRes.job_id ?? streamRes.jobId ?? null;
      const added = {
        job_id: jobId,
        camera_id: cameraId,
        title: newCamLocation || newCamName || `Camera ${cameraId}`,
        location: newCamLocation || "",
        status: jobId ? "LIVE" : "STARTED",
        videoUrl: null,
        frameSrc: null,
        latest_people: [],
        meta: { stream_url: newCamRtsp, is_stream: true }
      };
      setCameras(prev => [...prev, added]);
      setShowAddModal(false);
      setNewCamName("");
      setNewCamLocation("");
      setNewCamRtsp("");
    } catch (err) {
      setErrorMsg(err.message || "Failed to add camera");
    } finally {
      setAddingCamera(false);
    }
  };

  const handleStopStream = async (jobId) => {
    if (!jobId) return;
    try {
      const res = await fetch(`${API_BASE}/streams/${jobId}/stop`, { method: "POST" });
      if (res.ok) {
        setCameras(prev => prev.map(c => String(c.job_id) === String(jobId) ? ({ ...c, status: "STOPPED" }) : c));
        if (selectedCameraJobId && String(selectedCameraJobId) === String(jobId)) {
          setSelectedCameraJobId(null);
        }
      }
    } catch (e) {}
  };

  const nonExpandableStatuses = ["PROCESSING", "UPLOADING", "IDLE", "STARTED", "UPLOAD_FAILED", "STOPPED"];

  const selectedCamera = cameras.find(c => String(c.job_id) === String(selectedCameraJobId)) || null;

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
          <button
            className="px-4 py-3 bg-[#19325C] text-white rounded-lg hover:bg-[#5388DF] transition flex items-center gap-2"
            onClick={() => setShowAddModal(true)}
          >
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
            .map((camera, index) => {
              const isExpandable = !nonExpandableStatuses.includes(String(camera.status).toUpperCase());
              const displayedTitle = (camera.videoUrl && (!camera.meta || camera.meta.is_stream === false)) ? (camera.meta?.title || camera.title) : (camera.location || camera.title);
              return (
                <div
                  key={camera.job_id || camera.camera_id || index}
                  className={`relative rounded-lg ${isExpandable ? "cursor-pointer hover:scale-105 transition-transform" : "opacity-80"}`}
                  onClick={() => { if (isExpandable) handleCameraClick(camera); }}
                >
                  <ImageCard
                    image={camera.frameSrc || camera.videoUrl}
                    title={displayedTitle}
                    time={formattedTime}
                    status={camera.status}
                    onClick={isExpandable ? () => handleCameraClick(camera) : undefined}
                    actionIcons={[...(isExpandable ? [{ icon: <FiMaximize2 size={16} />, onClick: () => handleCameraClick(camera) }] : [])]}
                  />

                  {!isExpandable && (
                    <div className="absolute inset-0 z-40 bg-transparent cursor-not-allowed" onClick={(e)=>e.stopPropagation()} />
                  )}

                </div>
              );
            })}
        </div>
      </section>

      {selectedCamera && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-6">
          <div className="bg-[#111214] rounded-lg w-full max-w-4xl p-4">
            <div className="flex justify-between items-center mb-3">
              <h3 className="text-white font-semibold">{selectedCamera.title}</h3>
              
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-300">{selectedCamera.status}</span>
                <button className="text-gray-300 px-3 py-1 rounded bg-gray-800" onClick={() => setSelectedCameraJobId(null)}>Close</button>
                {selectedCamera.job_id && selectedCamera.status === "LIVE" && (
                  <button className="text-gray-300 px-3 py-1 rounded bg-red-700 ml-2" onClick={() => handleStopStream(selectedCamera.job_id)}>Stop Stream</button>
                )}
              </div>
            </div>
            <div className="w-full h-[60vh] bg-black rounded overflow-hidden flex items-center justify-center">
              {(() => {
                const isUploadedVideo = selectedCamera?.videoUrl && (selectedCamera?.meta?.is_stream === false || selectedCamera?.meta?.is_stream === "false");
                const hasAnnotatedFrame = !!selectedCamera?.frameSrc;

                if (isUploadedVideo) {
                  return (
                    <video
                      key={selectedCamera.videoUrl}
                      controls
                      playsInline
                      preload="metadata"
                      src={selectedCamera.videoUrl}
                      className="object-contain w-full h-full"
                    />
                  );
                }

                if (hasAnnotatedFrame) {
                  return <img alt="annotated" src={selectedCamera.frameSrc} className="object-contain w-full h-full" />;
                }

                if (selectedCamera?.videoUrl) {
                  return (
                    <video
                      key={selectedCamera.videoUrl}
                      controls
                      playsInline
                      preload="metadata"
                      src={selectedCamera.videoUrl}
                      className="object-contain w-full h-full"
                    />
                  );
                }

                return <div className="text-gray-400">No media available</div>;
              })()}
            </div>

            <div className="mt-3 text-xs text-gray-300">
              <pre className="text-xs text-gray-400 max-h-32 overflow-auto">{JSON.stringify(selectedCamera.latest_people || [], null, 2)}</pre>
            </div>
          </div>
        </div>
      )}

      {showAddModal && (
        <div className="fixed inset-0 z-60 flex items-center justify-center bg-black/60 p-4">
          <div className="bg-[#111214] rounded-lg w-full max-w-md p-6">
            <h3 className="text-white text-lg mb-4">Add Camera / Start Live Stream</h3>
            <form onSubmit={handleAddCameraSubmit} className="space-y-3">
              <div>
                <label className="block text-sm text-gray-300 mb-1">Name</label>
                <input value={newCamName} onChange={(e)=>setNewCamName(e.target.value)} className="w-full px-3 py-2 rounded bg-[#222227] text-gray-200 border border-gray-700" placeholder="Camera name" />
              </div>
              <div>
                <label className="block text-sm text-gray-300 mb-1">Location</label>
                <input value={newCamLocation} onChange={(e)=>setNewCamLocation(e.target.value)} className="w-full px-3 py-2 rounded bg-[#222227] text-gray-200 border border-gray-700" placeholder="Location (e.g. Warehouse entrance)" />
              </div>
              <div>
                <label className="block text-sm text-gray-300 mb-1">RTSP / Stream URL</label>
                <input value={newCamRtsp} onChange={(e)=>setNewCamRtsp(e.target.value)} required className="w-full px-3 py-2 rounded bg-[#222227] text-gray-200 border border-gray-700" placeholder="rtsp://user:pass@ip:554/stream or http://192.168.1.64:8080/video" />
              </div>
              {errorMsg && <div className="text-sm text-red-400">{errorMsg}</div>}
              <div className="flex justify-end gap-2 mt-2">
                <button type="button" className="px-3 py-2 rounded bg-gray-700 text-white" onClick={()=>{ setShowAddModal(false); setErrorMsg(""); }}>Cancel</button>
                <button type="submit" disabled={addingCamera} className="px-4 py-2 rounded bg-[#5388DF] text-white">
                  {addingCamera ? "Adding…" : "Add & Start Stream"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
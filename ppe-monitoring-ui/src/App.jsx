import { Routes, Route, Navigate } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Login from "./pages/Login";
import Camera from "./pages/Camera";
import Notifications from "./pages/Notifications";
import Incidents from "./pages/Incidents";
import Workers from "./pages/Workers";
import Reports from "./pages/Reports";

function App() {
  return (
    <Routes>
      {/* Login page */}
      <Route path="/" element={<Login />} />

      {/* Protected pages with sidebar */}
      <Route
        path="/*"
        element={
          <div className="flex">
            <Sidebar />
            <div className="flex-1 bg-[#FFFFFF] min-h-screen">
              <Routes>
                <Route path="camera" element={<Camera />} />
                <Route path="notifications" element={<Notifications />} />
                <Route path="incidents" element={<Incidents />} />
                <Route path="workers" element={<Workers />} />
                <Route path="reports" element={<Reports />} />
                {/* Default redirect to camera */}
                <Route path="*" element={<Navigate to="camera" replace />} />
              </Routes>
            </div>
          </div>
        }
      />
    </Routes>
  );
}

export default App;

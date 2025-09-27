import { Routes, Route, Navigate } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
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

      {/*pages with sidebar */}
      <Route
        path="/*"
        element={
          <div className="flex">
            <Sidebar />
            <div className="flex-1 bg-[#FFFFFF] min-h-screen">
              <Header />

              <div className="p-6">
                <Routes>
                  <Route path="camera" element={<Camera />} />
                  <Route path="notifications" element={<Notifications />} />
                  <Route path="incidents" element={<Incidents />} />
                  <Route path="workers" element={<Workers />} />
                  <Route path="reports" element={<Reports />} />
                  <Route path="*" element={<Navigate to="camera" replace />} />
                </Routes>
              </div>
            </div>
          </div>
        }
      />
    </Routes>
  );
}

export default App;

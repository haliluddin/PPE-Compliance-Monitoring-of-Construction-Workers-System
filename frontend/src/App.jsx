// frontend/src/App.jsx
import { Routes, Route, Navigate } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import Login from "./pages/Login";
import Camera from "./pages/Camera";
import Notifications from "./pages/Notifications";
import Incidents from "./pages/Incidents";
import Workers from "./pages/Workers";
import Reports from "./pages/Reports";
import WorkersProfile from "./pages/WorkersProfile";
import Register from "./pages/Register";
import ProtectedRoute from "./components/ProtectedRoute";
import { UnreadProvider } from "./context/UnreadContext";

function App() {
  return (
    <Routes>
      {/* Redirect root to login */}
      <Route path="/" element={<Navigate to="/login" replace />} />

      {/* Public routes */}
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />

      {/* Protected routes */}
      <Route
        path="/*"
        element={
          <ProtectedRoute>
            <div className="flex">
               <UnreadProvider>
              <Sidebar />
              <div className="flex-1 bg-[#1E1F23] min-h-screen">
                <Header />

                <div className="p-6 bg-[#1E1F23] rounded-lg m-4 text-gray-100">
                  <Routes>
                    <Route path="camera" element={<Camera />} />
                    <Route path="notifications" element={<Notifications />} />
                    <Route path="incidents" element={<Incidents />} />
                    <Route path="workers" element={<Workers />} />
                    <Route path="reports" element={<Reports />} />
                    <Route path="workersprofile/:id" element={<WorkersProfile />} />
                    <Route path="*" element={<Navigate to="camera" replace />} />
                  </Routes>
                </div>
              </div>
              </UnreadProvider>
            </div>
          </ProtectedRoute>
        }
      />
    </Routes>
  );
}

export default App;

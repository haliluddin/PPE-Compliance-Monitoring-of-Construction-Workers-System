import { Routes, Route, Navigate } from "react-router-dom";
import Login from "./pages/Login";
import Camera from "./pages/Camera";
import Notifications from "./pages/Notifications";
import Incidents from "./pages/Incidents";
import Workers from "./pages/Workers";
import Reports from "./pages/Reports";
import Layout from "./components/Layouts";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Login />} />

      {/* All sidebar pages use Layout */}
      <Route element={<Layout />}>
        <Route path="camera" element={<Camera />} />
        <Route path="notifications" element={<Notifications />} />
        <Route path="incidents" element={<Incidents />} />
        <Route path="workers" element={<Workers />} />
        <Route path="reports" element={<Reports />} />

        {/* default redirect */}
        <Route path="*" element={<Navigate to="camera" replace />} />
      </Route>
    </Routes>
  );
}

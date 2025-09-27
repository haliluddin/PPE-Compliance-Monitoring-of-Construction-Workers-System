import { FaEye, FaCheckCircle, FaExclamationCircle } from "react-icons/fa";

export default function Incident() {
 
  const notifications = [
    {
      id: 1,
      camera: "Camera A",
      violation: "No Helmet",
      status: "Pending",
      date: "2025-09-27",
    },
    {
      id: 2,
      camera: "Camera B",
      violation: "No Vest",
      status: "Resolved",
      date: "2025-09-26",
    },
  ];

  const statusBadge = (status) => {
    switch (status) {
      case "Resolved":
        return (
          <span className="inline-flex items-center gap-1 px-3 py-1 text-sm font-medium rounded-full bg-green-100 text-green-700">
            <FaCheckCircle />
            {status}
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-3 py-1 text-sm font-medium rounded-full bg-yellow-100 text-yellow-700">
            <FaExclamationCircle />
            {status}
          </span>
        );
    }
  };

  return (
    <div className="p-6 text-[#19325C]">

      {/* ---------- Filters ---------- */}
      <div className="flex gap-5 flex-wrap items-center">
        {["Camera", "Violations", "Status"].map((label) => (
          <div className="flex flex-col" key={label}>
            <label>
              <h1 className="text-md font-bold">{label}</h1>
            </label>
            <select className="px-3 py-2 border border-gray-300 rounded-lg bg-white text-[#19325C] shadow-sm focus:outline-none focus:ring-2 focus:ring-[#21005D]">
              <option>Select an option</option>
              <option>Placeholder</option>
              <option>Placeholder</option>
              <option>Placeholder</option>
            </select>
          </div>
        ))}
      </div>

      <div className="mt-10 overflow-x-auto rounded-xl border border-gray-200 shadow-sm">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-[#19325C] text-white">
            <tr>
              <th className="px-6 py-3 text-left text-sm font-semibold uppercase tracking-wider">
                Worker
              </th>
              <th className="px-6 py-3 text-left text-sm font-semibold uppercase tracking-wider">
                Camera Location
              </th>
              <th className="px-6 py-3 text-left text-sm font-semibold uppercase tracking-wider">
                Violation
              </th>
              <th className="px-6 py-3 text-left text-sm font-semibold uppercase tracking-wider">
                Status
              </th>
              <th className="px-6 py-3 text-left text-sm font-semibold uppercase tracking-wider">
                Date & Time
              </th>
              <th className="px-6 py-3 text-center text-sm font-semibold uppercase tracking-wider">
                Action
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 bg-white">
            {notifications.map((n) => (
              <tr
                key={n.id}
                className="hover:bg-gray-50 transition-colors duration-150"
              >
                <td className="px-6 py-4 text-sm font-medium">{n.id}</td>
                <td className="px-6 py-4 text-sm">{n.camera}</td>
                <td className="px-6 py-4 text-sm">{n.violation}</td>
                <td className="px-6 py-4 text-sm">{statusBadge(n.status)}</td>
                <td className="px-6 py-4 text-sm">{n.date}</td>
                <td className="px-6 py-4 text-center">
                  <button
                    className="inline-flex items-center px-3 py-2 text-sm font-medium text-white bg-[#19325C] rounded-lg hover:bg-[#152747] transition-colors"
                  >
                    <FaEye className="mr-2" />
                    View
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

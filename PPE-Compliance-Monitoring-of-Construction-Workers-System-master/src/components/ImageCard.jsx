import { FiMaximize2, FiSettings, FiWifiOff } from "react-icons/fi";

export default function ImageCard({ image, title, time, onClick, status, actionIcons, className, onRemove }) {
  return (
    <div
      onClick={onClick}
      className={`rounded-lg border border-gray-700 shadow-xl cursor-pointer hover:scale-105 hover:shadow-2xl transition-all group ${className}`}
    >
      {/* Image Container */}
      <div className="relative bg-[#2A2B30] rounded-lg overflow-hidden">
        <img src={image} alt={title} className="w-full h-48 object-cover rounded-lg" />
        {status === 'OFFLINE' && (
          <div className="absolute inset-0 bg-black bg-opacity-80 flex items-center justify-center text-white text-lg font-semibold">
            Unavailable
          </div>
        )}
        {status === 'NO SIGNAL' && (
          <div className="absolute inset-0 bg-black bg-opacity-95 flex flex-col items-center justify-center text-white text-lg font-semibold">
            <FiWifiOff size={48} className="mb-2" />
            
          </div>
        )}
        
        {/* Camera name overlay at top left */}
        <div className="absolute top-3 left-3 bg-black bg-opacity-50 px-3 py-1 rounded-md text-white">
          <h3 className="font-semibold text-sm">{title}</h3>
          
        </div>
        
        {/* Status badge */}
        {status && (
          <div className={`absolute top-3 right-3 text-white text-xs font-medium px-2 py-1 rounded ${status === 'LIVE' ? 'bg-green-500' : status === 'OFFLINE' ? 'bg-red-500' : 'bg-amber-500'}`}>
            {status}
          </div>
        )}
        {/* Time display at bottom left */}
        {status !== 'OFFLINE' && status !== 'NO SIGNAL' && (
          <div className="absolute bottom-3 left-3 bg-black bg-opacity-50 px-3 py-1 rounded-md text-left text-sm text-white">
            {time}
          </div>
        )}
        {/* Action icons at bottom right */}
        <div className="absolute bottom-3 right-3 flex justify-end gap-2">
          {actionIcons?.map((action, index) => {
            if (action.icon.type.name === "FiSettings") {
              return (
                <div key={index} className="relative group">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      // Toggle dropdown visibility
                      const dropdown = e.currentTarget.nextSibling;
                      if (dropdown) {
                        dropdown.classList.toggle('hidden');
                      }
                    }}
                    className="p-1.5 bg-gray-800 bg-opacity-70 hover:bg-gray-700 rounded-full text-gray-300 hover:text-white transition"
                  >
                    {action.icon}
                  </button>
                  <div className="absolute right-0 mt-2 w-48 bg-[#2A2B30] rounded-md shadow-lg py-1 z-10 hidden">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onRemove(); // Call the onRemove prop
                        // Hide dropdown after action
                        e.currentTarget.closest('.hidden').classList.add('hidden');
                      }}
                      className="block w-full text-left px-4 py-2 text-sm text-white hover:bg-gray-700"
                    >
                      Remove Camera
                    </button>
                    {/* Add more options here if needed */}
                  </div>
                </div>
              );
            } else {
              return (
                <button
                  key={index}
                  onClick={(e) => {
                    e.stopPropagation();
                    action.onClick();
                  }}
                  className="p-1.5 bg-gray-800 bg-opacity-70 hover:bg-gray-700 rounded-full text-gray-300 hover:text-white transition"
                >
                  {action.icon}
                </button>
              );
            }
          })}
        </div>
      </div>

    </div>
  );
}
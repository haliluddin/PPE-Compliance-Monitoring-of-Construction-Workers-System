import { FiUpload, FiCamera } from "react-icons/fi";
import ImageCard from "../components/ImageCard";
import { useState, useEffect } from "react";

export default function Camera() {

  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000); 

    return () => clearInterval(timer);
  }, []);

  // Format date and time 
  const formattedDate = currentTime.toLocaleDateString(); Date
  const formattedTime = currentTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); // Time


  return (
    <div className="p-8 text-[#19325C]">
      <h1 className="text-2xl font-bold mb-4">Camera List</h1>

      <div className="flex justify-between items-center">
          <select className="px-4 py-2 border border-gray-300 rounded-lg bg-white text-[#19325C] focus:outline-none focus:ring-2 focus:ring-[#21005D]">
            <option value="">Select an option</option>
            <option value="">Placeholder</option>
            <option value="">Placeholder</option>
            <option value="">Placeholder</option>
            <option value="">Placeholder</option>
            <option value="">Placeholder</option>
          </select>

          <div className="flex gap-4">
              {/* Upload Videos */}
              <button className="flex items-center gap-2 px-4 py-2 bg-[#5388DF] text-white rounded-lg hover:bg-[#19325C] transition">
                <FiUpload size={20} />
                Upload Videos
              </button>

              {/* Add Camera */}
              <button className="flex items-center gap-2 px-4 py-2 bg-blue text-white rounded-lg hover:bg-[#5388DF] transition">
                <FiCamera size={20} />
                Add Camera
              </button>
          </div>  
      </div>
      <div className="flex items-center gap-5 mt-5">
        <div className=" bg-[#21005D]/10 text-[#21005D] px-3 py-1 rounded-full">
          {formattedDate} - {formattedTime}
        </div>

        <div className="flex items-center gap-2 bg-[#21005D]/10 text-[#21005D] px-3 py-1 rounded-full">
          <div className="w-4 h-4 rounded-full bg-[#1db022]"></div>
          <span className="text-[#19325C] text-sm">Compliance</span>
        </div>

        <div className="flex items-center gap-2 bg-[#21005D]/10 text-[#21005D] px-3 py-1 rounded-full">
          <div className="w-4 h-4 rounded-full bg-[#d92323]"></div>
          <span className="text-[#19325C] text-sm">Non-Compliance</span>
        </div>
      </div>

      <div className="flex items-center gap-5 mt-10">
        <ImageCard
            image="https://images.unsplash.com/photo-1535379453347-1ffd615e2e08?auto=format&fit=crop&w=800&q=80"
            title="Camera 1"
            time="12:45 PM"
          />

      </div>
      
    </div>
  );
}

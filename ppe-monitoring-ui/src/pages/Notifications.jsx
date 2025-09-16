export default function Notifications() {
    return (
      <div className="p-6 text-[#19325C]">
        <h1 className="text-2xl font-bold mb-4">Notifications</h1>
        
        <div className="flex justify-between items-center mt-10">
            <div className="flex flex-col gap-3">
                <label htmlFor="camera"><h1 className="text-md font-bold">Camera</h1></label>
                <select className="px-4 py-2 border border-gray-300 rounded-lg bg-white text-[#19325C] focus:outline-none focus:ring-2 focus:ring-[#21005D]">
                    <option value="">Select an option</option>
                    <option value="">Placeholder</option>
                    <option value="">Placeholder</option>
                    <option value="">Placeholder</option>
                    <option value="">Placeholder</option>
                    <option value="">Placeholder</option>
                </select>
            </div>
            

        </div>
      </div>
    );
  }
  
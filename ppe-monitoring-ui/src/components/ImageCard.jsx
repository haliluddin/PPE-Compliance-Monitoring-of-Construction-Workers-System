export default function ImageCard({ image, title, time, onClick }) {
    return (
      <div
        onClick={onClick}
        className="w-64 bg-white rounded-lg border-2 border-blue shadow-xl overflow-hidden cursor-pointer hover:scale-105 hover:shadow-2xl transition-all"
      >
        {/* Image */}
        <div className="relative">
          <img src={image} alt={title} className="w-full h-40 object-cover" />
          
          {/* Overlay with title and time */}
          <div className="absolute bottom-0 left-0 right-0 bg-blue/70 text-white px-4 py-2 flex justify-between items-center">
            <span className="font-semibold">{title}</span>
            <span className="text-sm">{time}</span>
          </div>
        </div>
      </div>
    );
  }
  
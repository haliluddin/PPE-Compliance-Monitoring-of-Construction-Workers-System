export default function CustomTable({ columns, data, actions }) {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full bg-white rounded-lg shadow-md border border-gray-200">
        <thead className="bg-[#21005D] text-white">
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                className="text-left px-4 py-2 text-sm font-medium"
              >
                {col.label}
              </th>
            ))}
            {actions && <th className="px-4 py-2 text-sm font-medium">Actions</th>}
          </tr>
        </thead>

        <tbody className="text-[#19325C]">
          {data.map((row, index) => (
            <tr
              key={index}
              className="border-b last:border-b-0 hover:bg-[#5388DF]/10 transition"
            >
              {columns.map((col) => (
                <td key={col.key} className="px-4 py-2 text-sm">
                  {row[col.key]}
                </td>
              ))}
              {actions && (
                <td className="px-4 py-2 text-sm flex gap-2">
                  {actions.map((action) => (
                    <button
                      key={action.label}
                      onClick={() => action.onClick(row)}
                      className={`px-2 py-1 rounded-md text-white text-xs ${action.color}`}
                    >
                      {action.label}
                    </button>
                  ))}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

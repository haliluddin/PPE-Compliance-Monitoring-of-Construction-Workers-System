// frontend/src/context/UnreadContext.jsx
import { createContext, useState, useContext } from "react";

const UnreadContext = createContext();

export const UnreadProvider = ({ children }) => {
  const [unreadCount, setUnreadCount] = useState(0);
  return (
    <UnreadContext.Provider value={{ unreadCount, setUnreadCount }}>
      {children}
    </UnreadContext.Provider>
  );
};

export const useUnread = () => useContext(UnreadContext);

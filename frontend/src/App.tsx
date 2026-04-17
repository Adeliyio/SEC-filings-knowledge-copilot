import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import ChatPage from "./pages/ChatPage";
import AdminPage from "./pages/AdminPage";

const navStyle = (isActive: boolean) => ({
  padding: "8px 18px",
  borderRadius: 8,
  fontSize: 14,
  fontWeight: isActive ? 600 : 500,
  color: isActive ? "#4f46e5" : "#64748b",
  backgroundColor: isActive ? "#eef2ff" : "transparent",
  textDecoration: "none" as const,
  transition: "all 0.2s ease",
  letterSpacing: "-0.01em",
  fontFamily: "'Space Grotesk', sans-serif",
});

export default function App() {
  return (
    <BrowserRouter>
      <div style={{ minHeight: "100vh", backgroundColor: "#f8fafc", fontFamily: "'Space Grotesk', sans-serif" }}>
        {/* Top nav bar */}
        <nav
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "12px 28px",
            backgroundColor: "#ffffff",
            borderBottom: "1px solid #e2e8f0",
            boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div
              style={{
                width: 32,
                height: 32,
                borderRadius: 8,
                background: "linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#fff",
                fontSize: 16,
                fontWeight: 700,
              }}
            >
              T
            </div>
            <span style={{ fontSize: 17, fontWeight: 700, color: "#0f172a", letterSpacing: "-0.02em" }}>
              Tommy's Knowledge Copilot
            </span>
            <span
              style={{
                fontSize: 11,
                padding: "3px 8px",
                borderRadius: 6,
                background: "linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%)",
                color: "#4338ca",
                fontWeight: 600,
                letterSpacing: "0.02em",
              }}
            >
              SEC 10-K
            </span>
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            <NavLink to="/" style={({ isActive }) => navStyle(isActive)} end>
              Chat
            </NavLink>
            <NavLink to="/admin" style={({ isActive }) => navStyle(isActive)}>
              Admin
            </NavLink>
          </div>
        </nav>

        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/admin" element={<AdminPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

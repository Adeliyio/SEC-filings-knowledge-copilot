import type { ClaimInfo } from "../services/api";

interface Props {
  claims: ClaimInfo[];
}

const statusStyles: Record<string, { bg: string; border: string; text: string; dot: string; label: string }> = {
  supported: { bg: "#f0fdf4", border: "#bbf7d0", text: "#166534", dot: "#22c55e", label: "Supported" },
  partially_supported: { bg: "#fefce8", border: "#fde68a", text: "#854d0e", dot: "#eab308", label: "Partial" },
  unsupported: { bg: "#fef2f2", border: "#fecaca", text: "#991b1b", dot: "#ef4444", label: "Unsupported" },
};

export default function ClaimGrounding({ claims }: Props) {
  if (!claims.length) return null;

  const supported = claims.filter((c) => c.status === "supported").length;
  const partial = claims.filter((c) => c.status === "partially_supported").length;
  const unsupported = claims.filter((c) => c.status === "unsupported").length;

  return (
    <div style={{ marginTop: 12 }}>
      <div
        style={{
          fontSize: 12,
          fontWeight: 600,
          marginBottom: 8,
          color: "#475569",
          display: "flex",
          alignItems: "center",
          gap: 8,
          fontFamily: "'Space Grotesk', sans-serif",
        }}
      >
        Claim Verification
        <span style={{ fontWeight: 400, color: "#94a3b8" }}>
          {supported} supported, {partial} partial, {unsupported} unsupported
        </span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {claims.map((claim, i) => {
          const s = statusStyles[claim.status] || statusStyles.unsupported;
          return (
            <div
              key={i}
              style={{
                padding: "10px 12px",
                borderRadius: 8,
                backgroundColor: s.bg,
                border: `1px solid ${s.border}`,
                fontSize: 13,
              }}
            >
              <div style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
                <span
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 4,
                    fontSize: 10,
                    fontWeight: 600,
                    color: s.text,
                    textTransform: "uppercase",
                    letterSpacing: "0.04em",
                    flexShrink: 0,
                    marginTop: 2,
                    fontFamily: "'Space Grotesk', sans-serif",
                  }}
                >
                  <span
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: "50%",
                      backgroundColor: s.dot,
                    }}
                  />
                  {s.label}
                </span>
                <span style={{ color: "#1e293b", lineHeight: 1.5 }}>{claim.claim}</span>
              </div>
              {claim.evidence && (
                <div
                  style={{
                    marginTop: 6,
                    fontSize: 12,
                    color: "#64748b",
                    fontStyle: "italic",
                    paddingLeft: 20,
                    lineHeight: 1.5,
                  }}
                >
                  {claim.evidence}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

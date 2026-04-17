import { useState } from "react";
import type { SourceInfo } from "../services/api";

interface Props {
  sources: SourceInfo[];
}

export default function SourcePanel({ sources }: Props) {
  const [expanded, setExpanded] = useState<number | null>(null);

  if (!sources.length) return null;

  return (
    <div
      style={{
        marginTop: 12,
        border: "1px solid #e2e8f0",
        borderRadius: 10,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          padding: "8px 14px",
          backgroundColor: "#f8fafc",
          fontWeight: 600,
          fontSize: 12,
          color: "#475569",
          borderBottom: "1px solid #e2e8f0",
          letterSpacing: "0.01em",
          fontFamily: "'Space Grotesk', sans-serif",
        }}
      >
        Sources ({sources.length})
      </div>
      <div style={{ maxHeight: 300, overflowY: "auto" }}>
        {sources.map((source, i) => (
          <div
            key={source.chunk_id || i}
            style={{
              borderBottom: i < sources.length - 1 ? "1px solid #f1f5f9" : undefined,
            }}
          >
            <div
              onClick={() => setExpanded(expanded === i ? null : i)}
              style={{
                padding: "10px 14px",
                cursor: "pointer",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                fontSize: 13,
                color: "#1e293b",
                transition: "background-color 0.15s",
                fontFamily: "'Space Grotesk', sans-serif",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = "#f8fafc";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = "transparent";
              }}
            >
              <span>
                <strong style={{ color: "#6366f1" }}>[{source.index}]</strong>{" "}
                {source.company_name} &mdash;{" "}
                {source.section_path || source.source_file}
                {source.page_number != null && ` (p. ${source.page_number})`}
              </span>
              <span
                style={{
                  fontSize: 11,
                  color: "#94a3b8",
                  fontWeight: 500,
                }}
              >
                {(source.relevance_score * 100).toFixed(0)}%
              </span>
            </div>
            {expanded === i && (
              <div
                style={{
                  padding: "10px 14px",
                  backgroundColor: "#f8fafc",
                  fontSize: 12,
                  color: "#64748b",
                  whiteSpace: "pre-wrap",
                  lineHeight: 1.6,
                  borderTop: "1px solid #f1f5f9",
                }}
              >
                {source.text_preview}
                {source.is_table && (
                  <span
                    style={{
                      display: "inline-block",
                      marginTop: 6,
                      padding: "2px 8px",
                      background: "linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%)",
                      color: "#4338ca",
                      borderRadius: 4,
                      fontSize: 10,
                      fontWeight: 600,
                      letterSpacing: "0.03em",
                    }}
                  >
                    TABLE
                  </span>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

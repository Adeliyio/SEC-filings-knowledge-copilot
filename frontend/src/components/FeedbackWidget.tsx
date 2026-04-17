import { useState } from "react";
import { submitFeedback } from "../services/api";

interface Props {
  sessionId: string;
  messageId: string;
  query: string;
}

export default function FeedbackWidget({ sessionId, messageId, query }: Props) {
  const [submitted, setSubmitted] = useState<number | null>(null);

  const handleFeedback = async (rating: number) => {
    setSubmitted(rating);
    await submitFeedback(sessionId, messageId, query, rating);
  };

  if (submitted !== null) {
    return (
      <span style={{ fontSize: 12, color: "#94a3b8", fontWeight: 500 }}>
        {submitted === 5 ? "Thanks!" : "Thanks for feedback"}
      </span>
    );
  }

  return (
    <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
      <button
        onClick={() => handleFeedback(5)}
        style={{
          background: "none",
          border: "1px solid #e2e8f0",
          borderRadius: 6,
          padding: "3px 8px",
          cursor: "pointer",
          fontSize: 13,
          color: "#64748b",
          transition: "all 0.15s",
          fontFamily: "'Space Grotesk', sans-serif",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = "#dcfce7";
          e.currentTarget.style.borderColor = "#86efac";
          e.currentTarget.style.color = "#166534";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = "transparent";
          e.currentTarget.style.borderColor = "#e2e8f0";
          e.currentTarget.style.color = "#64748b";
        }}
        title="Helpful"
      >
        +
      </button>
      <button
        onClick={() => handleFeedback(1)}
        style={{
          background: "none",
          border: "1px solid #e2e8f0",
          borderRadius: 6,
          padding: "3px 8px",
          cursor: "pointer",
          fontSize: 13,
          color: "#64748b",
          transition: "all 0.15s",
          fontFamily: "'Space Grotesk', sans-serif",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = "#fee2e2";
          e.currentTarget.style.borderColor = "#fca5a5";
          e.currentTarget.style.color = "#991b1b";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = "transparent";
          e.currentTarget.style.borderColor = "#e2e8f0";
          e.currentTarget.style.color = "#64748b";
        }}
        title="Not helpful"
      >
        -
      </button>
    </div>
  );
}

import type { ClaimInfo, SourceInfo } from "../services/api";
import ClaimGrounding from "./ClaimGrounding";
import ConfidenceBadge from "./ConfidenceBadge";
import FeedbackWidget from "./FeedbackWidget";
import SourcePanel from "./SourcePanel";

export interface MessageData {
  id: string;
  role: "user" | "assistant";
  content: string;
  confidence?: number;
  sources?: SourceInfo[];
  claims?: ClaimInfo[];
  sessionId?: string;
  query?: string;
  status?: string;
  isStreaming?: boolean;
  attempts?: number;
}

interface Props {
  message: MessageData;
  showDetails: boolean;
  onToggleDetails: () => void;
}

export default function MessageBubble({ message, showDetails, onToggleDetails }: Props) {
  const isUser = message.role === "user";

  return (
    <div
      style={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        marginBottom: 18,
      }}
    >
      {/* Assistant avatar */}
      {!isUser && (
        <div
          style={{
            width: 30,
            height: 30,
            borderRadius: 8,
            background: "linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#fff",
            fontSize: 13,
            fontWeight: 700,
            marginRight: 10,
            flexShrink: 0,
            marginTop: 2,
          }}
        >
          T
        </div>
      )}

      <div
        style={{
          maxWidth: "78%",
          padding: isUser ? "10px 16px" : "14px 18px",
          borderRadius: isUser ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
          backgroundColor: isUser ? "#4f46e5" : "#ffffff",
          color: isUser ? "#ffffff" : "#1e293b",
          border: isUser ? "none" : "1px solid #e2e8f0",
          boxShadow: isUser
            ? "0 2px 8px rgba(79, 70, 229, 0.25)"
            : "0 1px 3px rgba(0,0,0,0.06)",
        }}
      >
        {/* Status indicator for streaming */}
        {message.status && message.isStreaming && (
          <div
            style={{
              fontSize: 12,
              color: isUser ? "rgba(255,255,255,0.7)" : "#818cf8",
              marginBottom: 8,
              fontWeight: 500,
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <span
              style={{
                display: "inline-block",
                width: 6,
                height: 6,
                borderRadius: "50%",
                backgroundColor: "#818cf8",
                animation: "blink 1.2s infinite",
              }}
            />
            {message.status}
          </div>
        )}

        {/* Message content */}
        <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.7, fontSize: 14, letterSpacing: "-0.01em" }}>
          {message.content}
          {message.isStreaming && (
            <span
              style={{
                display: "inline-block",
                width: 5,
                height: 16,
                backgroundColor: "#818cf8",
                marginLeft: 2,
                animation: "blink 1s infinite",
                verticalAlign: "text-bottom",
                borderRadius: 1,
              }}
            />
          )}
        </div>

        {/* Assistant metadata */}
        {!isUser && !message.isStreaming && message.content && (
          <div style={{ marginTop: 14, borderTop: "1px solid #f1f5f9", paddingTop: 10 }}>
            {/* Confidence + feedback row */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 12,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                {message.confidence != null && (
                  <ConfidenceBadge confidence={message.confidence} />
                )}
                {message.attempts != null && message.attempts > 1 && (
                  <span style={{ fontSize: 11, color: "#94a3b8", fontWeight: 500 }}>
                    {message.attempts} attempts
                  </span>
                )}
              </div>
              {message.sessionId && message.id && message.query && (
                <FeedbackWidget
                  sessionId={message.sessionId}
                  messageId={message.id}
                  query={message.query}
                />
              )}
            </div>

            {/* Toggle details */}
            {(message.sources?.length || message.claims?.length) ? (
              <button
                onClick={onToggleDetails}
                style={{
                  background: "none",
                  border: "none",
                  padding: 0,
                  marginTop: 10,
                  fontSize: 12,
                  color: "#6366f1",
                  cursor: "pointer",
                  fontWeight: 500,
                  fontFamily: "'Space Grotesk', sans-serif",
                  letterSpacing: "-0.01em",
                }}
              >
                {showDetails ? "Hide details" : "Show sources & verification"}
              </button>
            ) : null}

            {/* Expandable details */}
            {showDetails && (
              <>
                {message.sources && <SourcePanel sources={message.sources} />}
                {message.claims && <ClaimGrounding claims={message.claims} />}
              </>
            )}
          </div>
        )}
      </div>

      {/* User avatar */}
      {isUser && (
        <div
          style={{
            width: 30,
            height: 30,
            borderRadius: 8,
            backgroundColor: "#e2e8f0",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#475569",
            fontSize: 13,
            fontWeight: 600,
            marginLeft: 10,
            flexShrink: 0,
            marginTop: 2,
          }}
        >
          U
        </div>
      )}
    </div>
  );
}

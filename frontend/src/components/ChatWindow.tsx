import { useCallback, useEffect, useRef, useState } from "react";
import { chatStream } from "../services/api";
import type { ClaimInfo, SourceInfo } from "../services/api";
import MessageBubble, { type MessageData } from "./MessageBubble";

const FONT = "'Space Grotesk', sans-serif";

const sampleQuestions = [
  {
    icon: "AAPL",
    text: "What was Apple's total revenue in FY 2024?",
    color: "#6366f1",
  },
  {
    icon: "VS",
    text: "Compare R&D spending across Apple, Meta, and Microsoft",
    color: "#8b5cf6",
  },
  {
    icon: "META",
    text: "What regulatory risks does Meta disclose?",
    color: "#a855f7",
  },
  {
    icon: "MSFT",
    text: "What are Microsoft's remaining performance obligations?",
    color: "#7c3aed",
  },
];

export default function ChatWindow() {
  const [messages, setMessages] = useState<MessageData[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [expandedMessage, setExpandedMessage] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const query = input.trim();
      if (!query || isLoading) return;

      setInput("");
      setIsLoading(true);

      const userMsg: MessageData = {
        id: crypto.randomUUID(),
        role: "user",
        content: query,
      };

      const assistantId = crypto.randomUUID();
      const assistantMsg: MessageData = {
        id: assistantId,
        role: "assistant",
        content: "",
        isStreaming: true,
        status: "Starting...",
        query,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);

      let sources: SourceInfo[] = [];
      let claims: ClaimInfo[] = [];
      let confidence = 0;
      let attempts = 1;
      let finalSessionId = sessionId;

      try {
        await chatStream(query, sessionId, {
          onStatus: (phase, message) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, status: message } : m
              )
            );
          },
          onSources: (s) => {
            sources = s;
          },
          onToken: (token) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: m.content + token, status: "" }
                  : m
              )
            );
          },
          onVerification: (data) => {
            confidence = data.confidence;
            claims = data.claims;
            attempts = data.attempts;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      status: "Verification complete",
                      confidence,
                      claims,
                      attempts,
                    }
                  : m
              )
            );
          },
          onRetryAnswer: (answer, attempt) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      content: answer,
                      status: `Retry attempt ${attempt}`,
                    }
                  : m
              )
            );
          },
          onDone: (response) => {
            finalSessionId = response.session_id;
            setSessionId(response.session_id);
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      content: response.answer,
                      confidence: response.confidence,
                      sources: response.sources,
                      claims: response.claims,
                      sessionId: response.session_id,
                      attempts: response.attempts,
                      isStreaming: false,
                      status: undefined,
                    }
                  : m
              )
            );
          },
          onError: (error) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      content: `Error: ${error}`,
                      isStreaming: false,
                      status: undefined,
                    }
                  : m
              )
            );
          },
        });
      } catch (err) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content: `Error: ${err instanceof Error ? err.message : "Unknown error"}`,
                  isStreaming: false,
                  status: undefined,
                }
              : m
          )
        );
      }

      setIsLoading(false);
    },
    [input, isLoading, sessionId]
  );

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        maxWidth: 920,
        margin: "0 auto",
        fontFamily: FONT,
      }}
    >
      {/* Messages area */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "28px 28px 0",
          backgroundColor: "#f8fafc",
        }}
      >
        {messages.length === 0 && (
          <div
            style={{
              textAlign: "center",
              marginTop: 60,
            }}
          >
            <div
              style={{
                width: 56,
                height: 56,
                borderRadius: 16,
                background: "linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                margin: "0 auto 20px",
                boxShadow: "0 4px 14px rgba(79, 70, 229, 0.3)",
              }}
            >
              <span style={{ color: "#fff", fontSize: 24, fontWeight: 700 }}>T</span>
            </div>
            <h2
              style={{
                fontSize: 22,
                fontWeight: 700,
                color: "#0f172a",
                marginBottom: 6,
                letterSpacing: "-0.02em",
              }}
            >
              Tommy's Knowledge Copilot
            </h2>
            <p
              style={{
                fontSize: 15,
                color: "#64748b",
                marginBottom: 32,
                fontWeight: 400,
              }}
            >
              Ask Tommy about Apple, Meta, and Microsoft 10-K filings
            </p>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 12,
                maxWidth: 600,
                margin: "0 auto",
              }}
            >
              {sampleQuestions.map((q) => (
                <button
                  key={q.text}
                  onClick={() => setInput(q.text)}
                  style={{
                    background: "#ffffff",
                    border: "1px solid #e2e8f0",
                    borderRadius: 12,
                    padding: "14px 16px",
                    fontSize: 13,
                    color: "#334155",
                    cursor: "pointer",
                    textAlign: "left",
                    transition: "all 0.2s ease",
                    fontFamily: FONT,
                    lineHeight: 1.5,
                    boxShadow: "0 1px 2px rgba(0,0,0,0.04)",
                    display: "flex",
                    alignItems: "flex-start",
                    gap: 10,
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "#c7d2fe";
                    e.currentTarget.style.boxShadow = "0 2px 8px rgba(79, 70, 229, 0.1)";
                    e.currentTarget.style.transform = "translateY(-1px)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "#e2e8f0";
                    e.currentTarget.style.boxShadow = "0 1px 2px rgba(0,0,0,0.04)";
                    e.currentTarget.style.transform = "translateY(0)";
                  }}
                >
                  <span
                    style={{
                      flexShrink: 0,
                      fontSize: 10,
                      fontWeight: 700,
                      color: "#fff",
                      backgroundColor: q.color,
                      padding: "3px 6px",
                      borderRadius: 4,
                      letterSpacing: "0.03em",
                      marginTop: 1,
                    }}
                  >
                    {q.icon}
                  </span>
                  <span>{q.text}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble
            key={msg.id}
            message={msg}
            showDetails={expandedMessage === msg.id}
            onToggleDetails={() =>
              setExpandedMessage(expandedMessage === msg.id ? null : msg.id)
            }
          />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div
        style={{
          padding: "16px 28px 20px",
          backgroundColor: "#ffffff",
          borderTop: "1px solid #e2e8f0",
        }}
      >
        <form
          onSubmit={handleSubmit}
          style={{
            display: "flex",
            gap: 10,
            alignItems: "center",
            backgroundColor: "#f1f5f9",
            borderRadius: 14,
            padding: "6px 6px 6px 18px",
            border: "1px solid #e2e8f0",
            transition: "border-color 0.2s ease, box-shadow 0.2s ease",
          }}
          onFocus={(e) => {
            e.currentTarget.style.borderColor = "#a5b4fc";
            e.currentTarget.style.boxShadow = "0 0 0 3px rgba(99, 102, 241, 0.1)";
          }}
          onBlur={(e) => {
            e.currentTarget.style.borderColor = "#e2e8f0";
            e.currentTarget.style.boxShadow = "none";
          }}
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask Tommy about 10-K filings..."
            disabled={isLoading}
            style={{
              flex: 1,
              padding: "10px 0",
              border: "none",
              background: "transparent",
              fontSize: 14,
              outline: "none",
              fontFamily: FONT,
              color: "#0f172a",
              letterSpacing: "-0.01em",
            }}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            style={{
              padding: "10px 22px",
              borderRadius: 10,
              border: "none",
              background:
                isLoading || !input.trim()
                  ? "#cbd5e1"
                  : "linear-gradient(135deg, #4f46e5 0%, #6366f1 100%)",
              color: "#ffffff",
              fontWeight: 600,
              fontSize: 14,
              cursor: isLoading ? "not-allowed" : "pointer",
              fontFamily: FONT,
              letterSpacing: "-0.01em",
              transition: "all 0.2s ease",
              boxShadow:
                isLoading || !input.trim()
                  ? "none"
                  : "0 2px 8px rgba(79, 70, 229, 0.3)",
            }}
          >
            {isLoading ? "Thinking..." : "Send"}
          </button>
        </form>
        <p
          style={{
            textAlign: "center",
            fontSize: 11,
            color: "#94a3b8",
            marginTop: 8,
            letterSpacing: "0.01em",
          }}
        >
          Built by Tommy &middot; Powered by Llama 3.2 via Ollama &middot; Claim-level grounding
        </p>
      </div>

      <style>{`
        @keyframes blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }
      `}</style>
    </div>
  );
}

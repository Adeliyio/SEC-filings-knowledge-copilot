import { useCallback, useEffect, useState } from "react";
import type {
  DocumentInfo,
  EvalScore,
  FeedbackStats,
  IngestionStatus,
  SystemStats,
} from "../services/api";
import {
  fetchDocuments,
  fetchEvalScores,
  fetchEvalTrends,
  fetchFeedbackStats,
  fetchFeedbackTrends,
  fetchIngestionStatus,
  fetchSystemStats,
  triggerReindex,
  uploadDocument,
} from "../services/api";
import EvalDashboard from "../components/EvalDashboard";

type Tab = "overview" | "documents" | "evaluation" | "feedback";

export default function AdminPage() {
  const [tab, setTab] = useState<Tab>("overview");
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [ingestion, setIngestion] = useState<IngestionStatus | null>(null);
  const [evalScores, setEvalScores] = useState<EvalScore[]>([]);
  const [feedbackStats, setFeedbackStats] = useState<FeedbackStats | null>(null);
  const [evalTrends, setEvalTrends] = useState<any[]>([]);
  const [feedbackTrends, setFeedbackTrends] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [s, d, i, e, fs, et, ft] = await Promise.all([
        fetchSystemStats().catch(() => null),
        fetchDocuments().catch(() => ({ documents: [], total: 0 })),
        fetchIngestionStatus().catch(() => null),
        fetchEvalScores(20).catch(() => []),
        fetchFeedbackStats().catch(() => null),
        fetchEvalTrends().catch(() => []),
        fetchFeedbackTrends().catch(() => null),
      ]);
      setStats(s);
      setDocuments(d.documents);
      setIngestion(i);
      setEvalScores(e);
      setFeedbackStats(fs);
      setEvalTrends(et);
      setFeedbackTrends(ft);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleReindex = async () => {
    await triggerReindex();
    setTimeout(loadData, 2000);
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await uploadDocument(file);
    loadData();
  };

  const tabStyle = (t: Tab) => ({
    padding: "8px 20px",
    border: "none",
    borderBottom: tab === t ? "2px solid #2563eb" : "2px solid transparent",
    backgroundColor: "transparent",
    color: tab === t ? "#2563eb" : "#6b7280",
    fontWeight: tab === t ? 600 : 400,
    fontSize: 14,
    cursor: "pointer",
  });

  return (
    <div
      style={{
        maxWidth: 1100,
        margin: "0 auto",
        padding: "24px",
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      }}
    >
      {/* Tabs */}
      <div style={{ borderBottom: "1px solid #e5e7eb", marginBottom: 24, display: "flex", gap: 0 }}>
        <button onClick={() => setTab("overview")} style={tabStyle("overview")}>Overview</button>
        <button onClick={() => setTab("documents")} style={tabStyle("documents")}>Documents</button>
        <button onClick={() => setTab("evaluation")} style={tabStyle("evaluation")}>Evaluation</button>
        <button onClick={() => setTab("feedback")} style={tabStyle("feedback")}>Feedback</button>
      </div>

      {loading && <p style={{ color: "#9ca3af" }}>Loading...</p>}

      {/* Overview Tab */}
      {tab === "overview" && stats && (
        <div>
          <h2 style={{ margin: "0 0 16px", fontSize: 18, fontWeight: 700, color: "#111827" }}>
            System Overview
          </h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 24 }}>
            <StatCard label="Documents" value={stats.total_documents} />
            <StatCard label="Chunks" value={stats.total_chunks} />
            <StatCard label="Table Chunks" value={stats.total_table_chunks} />
            <StatCard label="Avg Chunks/Doc" value={stats.avg_chunks_per_doc} />
            <StatCard label="Eval Scores" value={stats.total_eval_scores} />
            <StatCard label="Feedback" value={stats.total_feedback} />
            <StatCard label="Provenance" value={stats.total_provenance} />
            <StatCard label="Companies" value={stats.companies.length} />
          </div>

          <div style={{ marginBottom: 24 }}>
            <h3 style={{ fontSize: 15, fontWeight: 600, color: "#374151", marginBottom: 8 }}>
              Indexed Companies
            </h3>
            <div style={{ display: "flex", gap: 8 }}>
              {stats.companies.map((c) => (
                <span
                  key={c}
                  style={{
                    padding: "4px 12px",
                    borderRadius: 16,
                    backgroundColor: "#eff6ff",
                    color: "#1e40af",
                    fontSize: 13,
                    fontWeight: 500,
                  }}
                >
                  {c}
                </span>
              ))}
            </div>
          </div>

          {/* Ingestion Status */}
          {ingestion && (
            <div
              style={{
                padding: 16,
                border: "1px solid #e5e7eb",
                borderRadius: 8,
                backgroundColor: "#fff",
                marginBottom: 24,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600, color: "#374151" }}>
                  Ingestion Pipeline
                </h3>
                <button
                  onClick={handleReindex}
                  disabled={ingestion.status === "running"}
                  style={{
                    padding: "6px 16px",
                    borderRadius: 6,
                    border: "none",
                    backgroundColor: ingestion.status === "running" ? "#9ca3af" : "#2563eb",
                    color: "#fff",
                    fontSize: 13,
                    fontWeight: 600,
                    cursor: ingestion.status === "running" ? "not-allowed" : "pointer",
                  }}
                >
                  {ingestion.status === "running" ? "Running..." : "Re-index All"}
                </button>
              </div>
              <div style={{ display: "flex", gap: 24, fontSize: 13, color: "#6b7280" }}>
                <span>
                  Status: <strong style={{ color: ingestion.status === "completed" ? "#16a34a" : ingestion.status === "failed" ? "#dc2626" : "#ca8a04" }}>{ingestion.status}</strong>
                </span>
                {ingestion.progress && <span>Progress: {ingestion.progress}</span>}
                {ingestion.files_processed > 0 && (
                  <span>Files: {ingestion.files_processed}/{ingestion.total_files}</span>
                )}
                {ingestion.error && <span style={{ color: "#dc2626" }}>Error: {ingestion.error}</span>}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Documents Tab */}
      {tab === "documents" && (
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
            <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: "#111827" }}>
              Documents ({documents.length})
            </h2>
            <label
              style={{
                padding: "6px 16px",
                borderRadius: 6,
                backgroundColor: "#2563eb",
                color: "#fff",
                fontSize: 13,
                fontWeight: 600,
                cursor: "pointer",
              }}
            >
              Upload Document
              <input type="file" accept=".pdf,.htm,.html" onChange={handleUpload} style={{ display: "none" }} />
            </label>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {documents.map((doc) => (
              <div
                key={doc.id}
                style={{
                  padding: 16,
                  border: "1px solid #e5e7eb",
                  borderRadius: 8,
                  backgroundColor: "#fff",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <div>
                    <strong style={{ fontSize: 15, color: "#111827" }}>{doc.source_file}</strong>
                    <span
                      style={{
                        marginLeft: 8,
                        padding: "2px 8px",
                        borderRadius: 4,
                        backgroundColor: "#f3f4f6",
                        fontSize: 12,
                        color: "#6b7280",
                      }}
                    >
                      {doc.file_format.toUpperCase()}
                    </span>
                  </div>
                  <span style={{ fontSize: 13, color: "#6b7280" }}>
                    {doc.chunk_count} chunks
                  </span>
                </div>
                <div style={{ display: "flex", gap: 16, fontSize: 13, color: "#6b7280" }}>
                  <span>Company: <strong style={{ color: "#374151" }}>{doc.company_name}</strong></span>
                  <span>Filing: {doc.filing_type}</span>
                  {doc.fiscal_year && <span>FY {doc.fiscal_year}</span>}
                  <span>{(doc.file_size_bytes / 1024).toFixed(0)} KB</span>
                  {doc.total_pages && <span>{doc.total_pages} pages</span>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Evaluation Tab */}
      {tab === "evaluation" && (
        <EvalDashboard scores={evalScores} trends={evalTrends} />
      )}

      {/* Feedback Tab */}
      {tab === "feedback" && (
        <div>
          <h2 style={{ margin: "0 0 16px", fontSize: 18, fontWeight: 700, color: "#111827" }}>
            Feedback Analytics
          </h2>

          {!feedbackStats || feedbackStats.total_feedback === 0 ? (
            <div
              style={{
                padding: 32,
                border: "1px dashed #cbd5e1",
                borderRadius: 8,
                backgroundColor: "#f8fafc",
                textAlign: "center",
              }}
            >
              <div style={{ fontSize: 15, fontWeight: 600, color: "#334155", marginBottom: 6 }}>
                No feedback collected yet
              </div>
              <div style={{ fontSize: 13, color: "#64748b", maxWidth: 520, margin: "0 auto" }}>
                User ratings from the chat interface appear here. Ask a question on the
                Chat page and click the <strong>+</strong> (helpful) or <strong>−</strong>{" "}
                (not helpful) button below any response to populate this dashboard.
              </div>
            </div>
          ) : (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 24 }}>
                <StatCard label="Total Feedback" value={feedbackStats.total_feedback} />
                <StatCard label="Positive" value={feedbackStats.positive_count} color="#16a34a" />
                <StatCard label="Negative" value={feedbackStats.negative_count} color="#dc2626" />
                <StatCard label="Avg Rating" value={feedbackStats.avg_rating.toFixed(1)} />
              </div>

              {/* Satisfaction bar */}
              <div style={{ marginBottom: 24 }}>
                <h3 style={{ fontSize: 15, fontWeight: 600, color: "#374151", marginBottom: 8 }}>
                  Satisfaction Distribution
                </h3>
                <div style={{ display: "flex", height: 24, borderRadius: 6, overflow: "hidden", backgroundColor: "#f3f4f6" }}>
                  <div
                    style={{
                      width: `${feedbackStats.positive_rate * 100}%`,
                      backgroundColor: "#16a34a",
                    }}
                  />
                  <div
                    style={{
                      width: `${(feedbackStats.neutral_count / feedbackStats.total_feedback) * 100}%`,
                      backgroundColor: "#ca8a04",
                    }}
                  />
                  <div
                    style={{
                      width: `${feedbackStats.negative_rate * 100}%`,
                      backgroundColor: "#dc2626",
                    }}
                  />
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#6b7280", marginTop: 4 }}>
                  <span>Positive ({(feedbackStats.positive_rate * 100).toFixed(0)}%)</span>
                  <span>Negative ({(feedbackStats.negative_rate * 100).toFixed(0)}%)</span>
                </div>
              </div>

              {/* Feedback trends chart */}
              {feedbackTrends && feedbackTrends.data_points && feedbackTrends.data_points.length > 0 && (
                <div style={{ marginBottom: 24 }}>
                  <h3 style={{ fontSize: 15, fontWeight: 600, color: "#374151", marginBottom: 8 }}>
                    Feedback Over Time
                  </h3>
                  <FeedbackChart data={feedbackTrends.data_points} />
                </div>
              )}

              {/* Recent negative feedback */}
              {feedbackStats.recent_negative.length > 0 && (
                <div>
                  <h3 style={{ fontSize: 15, fontWeight: 600, color: "#374151", marginBottom: 8 }}>
                    Recent Negative Feedback
                  </h3>
                  {feedbackStats.recent_negative.map((fb) => (
                    <div
                      key={fb.id}
                      style={{
                        padding: 12,
                        border: "1px solid #fee2e2",
                        borderRadius: 8,
                        backgroundColor: "#fef2f2",
                        marginBottom: 8,
                      }}
                    >
                      <div style={{ fontSize: 13, color: "#991b1b", marginBottom: 4 }}>
                        Rating: {fb.rating}/5 — {fb.created_at ? new Date(fb.created_at).toLocaleDateString() : ""}
                      </div>
                      <div style={{ fontSize: 14, color: "#1f2937" }}>{fb.query}</div>
                      {fb.comment && (
                        <div style={{ fontSize: 13, color: "#6b7280", marginTop: 4, fontStyle: "italic" }}>
                          "{fb.comment}"
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

// --- Sub-Components ---

function StatCard({ label, value, color }: { label: string; value: number | string; color?: string }) {
  return (
    <div
      style={{
        padding: 16,
        border: "1px solid #e5e7eb",
        borderRadius: 8,
        backgroundColor: "#fff",
        textAlign: "center",
      }}
    >
      <div style={{ fontSize: 24, fontWeight: 700, color: color || "#111827" }}>
        {typeof value === "number" ? value.toLocaleString() : value}
      </div>
      <div style={{ fontSize: 13, color: "#6b7280", marginTop: 4 }}>{label}</div>
    </div>
  );
}

function FeedbackChart({ data }: { data: { date: string; count: number; avg_rating: number; positive: number; negative: number }[] }) {
  const maxCount = Math.max(...data.map((d) => d.count), 1);

  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 4, height: 120, padding: "0 0 24px" }}>
      {data.slice(-30).map((d, i) => (
        <div
          key={i}
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 2,
          }}
        >
          <div
            style={{
              width: "100%",
              height: `${(d.count / maxCount) * 80}px`,
              backgroundColor: d.avg_rating >= 3.5 ? "#16a34a" : d.avg_rating >= 2.5 ? "#ca8a04" : "#dc2626",
              borderRadius: "2px 2px 0 0",
              minHeight: 2,
            }}
          />
          <div style={{ fontSize: 9, color: "#9ca3af", transform: "rotate(-45deg)", whiteSpace: "nowrap" }}>
            {d.date.slice(5)}
          </div>
        </div>
      ))}
    </div>
  );
}

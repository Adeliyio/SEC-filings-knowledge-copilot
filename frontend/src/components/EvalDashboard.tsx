import { useState } from "react";
import type { EvalScore } from "../services/api";

interface Props {
  scores: EvalScore[];
  trends: {
    date: string;
    count: number;
    avg_overall: number | null;
    avg_faithfulness: number | null;
    avg_answer_relevancy?: number | null;
    avg_factual_grounding?: number | null;
    avg_completeness?: number | null;
    avg_citation_quality?: number | null;
    avg_coherence?: number | null;
  }[];
}

export default function EvalDashboard({ scores, trends }: Props) {
  const [selectedScore, setSelectedScore] = useState<EvalScore | null>(null);

  // Compute summary stats from scores
  const avgOverall =
    scores.length > 0
      ? scores.reduce((s, e) => s + (e.overall_score || 0), 0) / scores.length
      : 0;
  const avgFaithfulness =
    scores.length > 0
      ? scores.reduce((s, e) => s + (e.faithfulness || 0), 0) /
        scores.filter((e) => e.faithfulness != null).length || 0
      : 0;
  const avgGrounding =
    scores.length > 0
      ? scores.reduce((s, e) => s + (e.factual_grounding || 0), 0) /
        scores.filter((e) => e.factual_grounding != null).length || 0
      : 0;

  return (
    <div>
      <h2
        style={{
          margin: "0 0 16px",
          fontSize: 18,
          fontWeight: 700,
          color: "#111827",
        }}
      >
        Evaluation Dashboard
      </h2>

      {/* Summary metrics */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: 16,
          marginBottom: 24,
        }}
      >
        <MetricCard label="Avg Overall Score" value={avgOverall} />
        <MetricCard label="Avg Faithfulness" value={avgFaithfulness} />
        <MetricCard label="Avg Factual Grounding" value={avgGrounding} />
        <MetricCard label="Total Evaluations" value={scores.length} isCount />
      </div>

      {/* Trend chart */}
      {trends.length > 0 && (
        <div
          style={{
            marginBottom: 24,
            padding: 16,
            border: "1px solid #e5e7eb",
            borderRadius: 8,
            backgroundColor: "#fff",
          }}
        >
          <h3
            style={{
              margin: "0 0 12px",
              fontSize: 15,
              fontWeight: 600,
              color: "#374151",
            }}
          >
            Score Trends (Daily)
          </h3>
          <TrendChart data={trends} />
        </div>
      )}

      {/* Score table */}
      <div
        style={{
          border: "1px solid #e5e7eb",
          borderRadius: 8,
          backgroundColor: "#fff",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            padding: "12px 16px",
            borderBottom: "1px solid #e5e7eb",
            fontWeight: 600,
            fontSize: 15,
            color: "#374151",
          }}
        >
          Recent Evaluations ({scores.length})
        </div>
        <div style={{ overflowX: "auto" }}>
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              fontSize: 13,
            }}
          >
            <thead>
              <tr style={{ backgroundColor: "#f9fafb", textAlign: "left" }}>
                <th style={thStyle}>Query</th>
                <th style={thStyle}>Overall</th>
                <th style={thStyle}>Faithful.</th>
                <th style={thStyle}>Relevancy</th>
                <th style={thStyle}>Grounding</th>
                <th style={thStyle}>Complete.</th>
                <th style={thStyle}>Citations</th>
                <th style={thStyle}>Coherence</th>
                <th style={thStyle}>Date</th>
              </tr>
            </thead>
            <tbody>
              {scores.map((score) => (
                <tr
                  key={score.id}
                  onClick={() =>
                    setSelectedScore(
                      selectedScore?.id === score.id ? null : score
                    )
                  }
                  style={{
                    cursor: "pointer",
                    borderBottom: "1px solid #f3f4f6",
                    backgroundColor:
                      selectedScore?.id === score.id ? "#eff6ff" : undefined,
                  }}
                >
                  <td style={tdStyle}>
                    {score.query.length > 50
                      ? score.query.slice(0, 50) + "..."
                      : score.query}
                  </td>
                  <td style={tdStyle}>
                    <ScoreBadge value={score.overall_score} />
                  </td>
                  <td style={tdStyle}>
                    <ScoreBadge value={score.faithfulness} />
                  </td>
                  <td style={tdStyle}>
                    <ScoreBadge value={score.answer_relevancy} />
                  </td>
                  <td style={tdStyle}>
                    <ScoreBadge value={score.factual_grounding} />
                  </td>
                  <td style={tdStyle}>
                    <ScoreBadge value={score.completeness} />
                  </td>
                  <td style={tdStyle}>
                    <ScoreBadge value={score.citation_quality} />
                  </td>
                  <td style={tdStyle}>
                    <ScoreBadge value={score.coherence} />
                  </td>
                  <td style={tdStyle}>
                    {score.created_at
                      ? new Date(score.created_at).toLocaleDateString()
                      : "-"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {scores.length === 0 && (
          <div
            style={{
              padding: 32,
              textAlign: "center",
              color: "#9ca3af",
              fontSize: 14,
            }}
          >
            No evaluation scores yet. Run queries or the golden dataset to
            generate scores.
          </div>
        )}
      </div>

      {/* Detail panel */}
      {selectedScore && (
        <div
          style={{
            marginTop: 16,
            padding: 16,
            border: "1px solid #e5e7eb",
            borderRadius: 8,
            backgroundColor: "#fff",
          }}
        >
          <h3
            style={{
              margin: "0 0 12px",
              fontSize: 15,
              fontWeight: 600,
              color: "#374151",
            }}
          >
            Score Details
          </h3>
          <div style={{ fontSize: 14, color: "#1f2937", marginBottom: 12 }}>
            <strong>Query:</strong> {selectedScore.query}
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: 12,
            }}
          >
            <DetailMetric
              label="Faithfulness"
              value={selectedScore.faithfulness}
            />
            <DetailMetric
              label="Answer Relevancy"
              value={selectedScore.answer_relevancy}
            />
            <DetailMetric
              label="Context Precision"
              value={selectedScore.context_precision}
            />
            <DetailMetric
              label="Context Recall"
              value={selectedScore.context_recall}
            />
            <DetailMetric
              label="Factual Grounding"
              value={selectedScore.factual_grounding}
            />
            <DetailMetric
              label="Completeness"
              value={selectedScore.completeness}
            />
            <DetailMetric
              label="Citation Quality"
              value={selectedScore.citation_quality}
            />
            <DetailMetric label="Coherence" value={selectedScore.coherence} />
          </div>
        </div>
      )}
    </div>
  );
}

// --- Sub-components ---

const thStyle: React.CSSProperties = {
  padding: "8px 12px",
  fontWeight: 600,
  fontSize: 12,
  color: "#6b7280",
  textTransform: "uppercase",
  letterSpacing: "0.05em",
};

const tdStyle: React.CSSProperties = {
  padding: "8px 12px",
  color: "#1f2937",
};

function ScoreBadge({ value }: { value: number | null }) {
  if (value == null) return <span style={{ color: "#d1d5db" }}>-</span>;

  let color: string;
  if (value >= 0.8) color = "#16a34a";
  else if (value >= 0.6) color = "#ca8a04";
  else color = "#dc2626";

  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 6px",
        borderRadius: 4,
        fontSize: 12,
        fontWeight: 600,
        color: "#fff",
        backgroundColor: color,
        minWidth: 36,
        textAlign: "center",
      }}
    >
      {(value * 100).toFixed(0)}
    </span>
  );
}

function MetricCard({
  label,
  value,
  isCount,
}: {
  label: string;
  value: number;
  isCount?: boolean;
}) {
  const display = isCount ? value.toString() : `${(value * 100).toFixed(0)}%`;
  let color = "#111827";
  if (!isCount) {
    if (value >= 0.8) color = "#16a34a";
    else if (value >= 0.6) color = "#ca8a04";
    else color = "#dc2626";
  }

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
      <div style={{ fontSize: 28, fontWeight: 700, color }}>{display}</div>
      <div style={{ fontSize: 13, color: "#6b7280", marginTop: 4 }}>
        {label}
      </div>
    </div>
  );
}

function DetailMetric({
  label,
  value,
}: {
  label: string;
  value: number | null;
}) {
  return (
    <div>
      <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div
          style={{
            flex: 1,
            height: 8,
            backgroundColor: "#f3f4f6",
            borderRadius: 4,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: `${(value || 0) * 100}%`,
              height: "100%",
              backgroundColor:
                (value || 0) >= 0.8
                  ? "#16a34a"
                  : (value || 0) >= 0.6
                    ? "#ca8a04"
                    : "#dc2626",
              borderRadius: 4,
            }}
          />
        </div>
        <span style={{ fontSize: 13, fontWeight: 600, color: "#374151", minWidth: 36 }}>
          {value != null ? `${(value * 100).toFixed(0)}%` : "-"}
        </span>
      </div>
    </div>
  );
}

function TrendChart({
  data,
}: {
  data: {
    date: string;
    count: number;
    avg_overall: number | null;
    avg_faithfulness?: number | null;
  }[];
}) {
  if (!data.length) return null;

  const points = data.filter((d) => d.avg_overall != null);
  if (!points.length) return null;

  const chartWidth = 700;
  const chartHeight = 120;
  const padding = { left: 40, right: 16, top: 8, bottom: 24 };
  const w = chartWidth - padding.left - padding.right;
  const h = chartHeight - padding.top - padding.bottom;

  const xStep = w / Math.max(points.length - 1, 1);

  const pathPoints = points.map((d, i) => {
    const x = padding.left + i * xStep;
    const y = padding.top + h - (d.avg_overall || 0) * h;
    return `${x},${y}`;
  });

  return (
    <svg
      viewBox={`0 0 ${chartWidth} ${chartHeight}`}
      style={{ width: "100%", height: chartHeight }}
    >
      {/* Y-axis labels */}
      {[0, 0.5, 1.0].map((v) => (
        <g key={v}>
          <text
            x={padding.left - 8}
            y={padding.top + h - v * h + 4}
            textAnchor="end"
            fontSize={10}
            fill="#9ca3af"
          >
            {(v * 100).toFixed(0)}
          </text>
          <line
            x1={padding.left}
            y1={padding.top + h - v * h}
            x2={chartWidth - padding.right}
            y2={padding.top + h - v * h}
            stroke="#f3f4f6"
            strokeWidth={1}
          />
        </g>
      ))}

      {/* Line */}
      <polyline
        points={pathPoints.join(" ")}
        fill="none"
        stroke="#2563eb"
        strokeWidth={2}
      />

      {/* Dots */}
      {points.map((d, i) => (
        <circle
          key={i}
          cx={padding.left + i * xStep}
          cy={padding.top + h - (d.avg_overall || 0) * h}
          r={3}
          fill="#2563eb"
        />
      ))}

      {/* X-axis labels (show every few) */}
      {points
        .filter((_, i) => i % Math.max(Math.floor(points.length / 6), 1) === 0)
        .map((d, i, arr) => {
          const idx = points.indexOf(d);
          return (
            <text
              key={i}
              x={padding.left + idx * xStep}
              y={chartHeight - 4}
              textAnchor="middle"
              fontSize={10}
              fill="#9ca3af"
            >
              {d.date.slice(5)}
            </text>
          );
        })}
    </svg>
  );
}

interface Props {
  confidence: number;
}

export default function ConfidenceBadge({ confidence }: Props) {
  const pct = Math.round(confidence * 100);
  let bg: string;
  let text: string;
  let label: string;

  if (confidence >= 0.8) {
    bg = "#dcfce7";
    text = "#166534";
    label = "High";
  } else if (confidence >= 0.65) {
    bg = "#fef9c3";
    text = "#854d0e";
    label = "Medium";
  } else {
    bg = "#fee2e2";
    text = "#991b1b";
    label = "Low";
  }

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        padding: "3px 10px",
        borderRadius: 20,
        fontSize: 11,
        fontWeight: 600,
        color: text,
        backgroundColor: bg,
        letterSpacing: "0.01em",
        fontFamily: "'Space Grotesk', sans-serif",
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          backgroundColor: text,
          opacity: 0.6,
        }}
      />
      {label} {pct}%
    </span>
  );
}

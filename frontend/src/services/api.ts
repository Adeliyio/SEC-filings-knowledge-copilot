const API_BASE = "/api";

// --- Chat Types ---

export interface SourceInfo {
  index: number;
  chunk_id: string;
  source_file: string;
  company_name: string;
  section_path: string;
  page_number: number | null;
  is_table: boolean;
  relevance_score: number;
  text_preview: string;
}

export interface ClaimInfo {
  claim: string;
  status: "supported" | "partially_supported" | "unsupported";
  evidence: string;
  reasoning: string;
}

export interface ChatResponse {
  answer: string;
  confidence: number;
  sources: SourceInfo[];
  claims: ClaimInfo[];
  session_id: string;
  message_id: string;
  attempts: number;
}

export interface SSECallbacks {
  onStatus?: (phase: string, message: string) => void;
  onPlan?: (plan: { strategy: string; sub_queries: { query: string; company_filter: string | null }[] }) => void;
  onSources?: (sources: SourceInfo[]) => void;
  onToken?: (token: string) => void;
  onVerification?: (data: { confidence: number; claims: ClaimInfo[]; attempts: number }) => void;
  onRetryAnswer?: (answer: string, attempt: number) => void;
  onDone?: (response: ChatResponse) => void;
  onError?: (error: string) => void;
}

// --- Admin Types ---

export interface DocumentInfo {
  id: string;
  source_file: string;
  company_name: string;
  filing_type: string;
  fiscal_year: number | null;
  file_format: string;
  file_size_bytes: number;
  total_pages: number | null;
  chunk_count: number;
  ingested_at: string | null;
}

export interface SystemStats {
  total_documents: number;
  total_chunks: number;
  total_table_chunks: number;
  total_feedback: number;
  total_eval_scores: number;
  total_provenance: number;
  companies: string[];
  avg_chunks_per_doc: number;
}

export interface EvalScore {
  id: number;
  query: string;
  response_id: string;
  faithfulness: number | null;
  answer_relevancy: number | null;
  context_precision: number | null;
  context_recall: number | null;
  factual_grounding: number | null;
  completeness: number | null;
  citation_quality: number | null;
  coherence: number | null;
  overall_score: number | null;
  created_at: string | null;
}

export interface FeedbackStats {
  total_feedback: number;
  positive_count: number;
  negative_count: number;
  neutral_count: number;
  avg_rating: number;
  positive_rate: number;
  negative_rate: number;
  recent_negative: { id: string; query: string; rating: number; comment: string; created_at: string }[];
}

export interface FeedbackTrend {
  period: string;
  data_points: { date: string; count: number; avg_rating: number; positive: number; negative: number }[];
}

export interface IngestionStatus {
  status: string;
  started_at: string | null;
  completed_at: string | null;
  progress: string;
  files_processed: number;
  total_files: number;
  error: string | null;
}

// --- Chat API ---

export async function chatStream(
  query: string,
  sessionId: string | null,
  callbacks: SSECallbacks
): Promise<void> {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, session_id: sessionId }),
  });

  if (!response.ok) {
    callbacks.onError?.(`HTTP ${response.status}: ${response.statusText}`);
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    callbacks.onError?.("No response body");
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    let currentEvent = "";
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith("data: ") && currentEvent) {
        try {
          const data = JSON.parse(line.slice(6));
          switch (currentEvent) {
            case "status":
              callbacks.onStatus?.(data.phase, data.message);
              break;
            case "plan":
              callbacks.onPlan?.(data);
              break;
            case "sources":
              callbacks.onSources?.(data.sources);
              break;
            case "token":
              callbacks.onToken?.(data.token);
              break;
            case "verification":
              callbacks.onVerification?.(data);
              break;
            case "retry_answer":
              callbacks.onRetryAnswer?.(data.answer, data.attempt);
              break;
            case "done":
              callbacks.onDone?.(data);
              break;
          }
        } catch {
          // skip malformed JSON
        }
        currentEvent = "";
      }
    }
  }
}

export async function submitFeedback(
  sessionId: string,
  messageId: string,
  query: string,
  rating: number,
  comment: string = ""
): Promise<void> {
  await fetch(`${API_BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      message_id: messageId,
      query,
      rating,
      comment,
    }),
  });
}

// --- Admin API ---

export async function fetchSystemStats(): Promise<SystemStats> {
  const res = await fetch(`${API_BASE}/admin/stats`);
  return res.json();
}

export async function fetchDocuments(): Promise<{ documents: DocumentInfo[]; total: number }> {
  const res = await fetch(`${API_BASE}/documents`);
  return res.json();
}

export async function fetchIngestionStatus(): Promise<IngestionStatus> {
  const res = await fetch(`${API_BASE}/admin/ingestion/status`);
  return res.json();
}

export async function triggerReindex(file?: string): Promise<{ status: string; message: string }> {
  const url = file ? `${API_BASE}/admin/reindex?file=${encodeURIComponent(file)}` : `${API_BASE}/admin/reindex`;
  const res = await fetch(url, { method: "POST" });
  return res.json();
}

export async function uploadDocument(file: File): Promise<{ status: string; filename: string; message: string }> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/documents/upload`, { method: "POST", body: formData });
  return res.json();
}

export async function fetchDataFiles(): Promise<{ data_dir: string; files: { filename: string; size_bytes: number; size_kb: number; format: string }[] }> {
  const res = await fetch(`${API_BASE}/admin/data-files`);
  return res.json();
}

// --- Eval API ---

export async function fetchEvalScores(limit = 50): Promise<EvalScore[]> {
  const res = await fetch(`${API_BASE}/eval/scores?limit=${limit}`);
  return res.json();
}

export async function fetchEvalTrends(days = 30): Promise<{ date: string; count: number; avg_overall: number | null; avg_faithfulness: number | null }[]> {
  const res = await fetch(`${API_BASE}/eval/scores/trends?days=${days}`);
  return res.json();
}

export async function fetchFeedbackStats(days = 30): Promise<FeedbackStats> {
  const res = await fetch(`${API_BASE}/eval/feedback/stats?days=${days}`);
  return res.json();
}

export async function fetchFeedbackTrends(period = "daily", days = 30): Promise<FeedbackTrend> {
  const res = await fetch(`${API_BASE}/eval/feedback/trends?period=${period}&days=${days}`);
  return res.json();
}

export async function fetchGoldenDataset(): Promise<{ total_entries: number; categories: Record<string, number>; entries: { id: string; query: string; category: string; company: string | null; difficulty: string }[] }> {
  const res = await fetch(`${API_BASE}/eval/golden-dataset`);
  return res.json();
}

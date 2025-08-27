import React, { useMemo, useState } from "react";

type Step = { id: number; title: string; note: string };

function tokenizeSimple(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9_]+/g) ?? []);
}

function scoreProxyAttention(query: string, notes: string[], finalAnswer: string): Record<string, number> {
  const faTokens = tokenizeSimple(finalAnswer);
  const base: Record<string, number> = {};
  for (const t of faTokens) base[t] = (base[t] ?? 0) + 1;
  const mx = Math.max(1, ...Object.values(base));
  for (const k of Object.keys(base)) base[k] = base[k] / mx;

  const scores: Record<string, number> = {};
  for (const tok of tokenizeSimple([query, ...notes].join(" "))) {
    scores[tok] = Math.max(scores[tok] ?? 0, base[tok] ?? 0);
  }
  return scores;
}

function mockPlan(task: string): Step[] {
  // Lightweight “planner” to mimic the desktop app
  const skeleton = [
    { id: 1, title: "Fact retrieval" },
    { id: 2, title: "Context mapping" },
    { id: 3, title: "Reasoning" },
    { id: 4, title: "Synthesis" },
  ];
  return skeleton.map(s => ({
    ...s,
    note: `High-level summary for “${s.title}” in the context of: ${task.slice(0, 80)}${task.length > 80 ? "…" : ""}`
  }));
}

export default function App() {
  const [task, setTask] = useState("Explain transformers at a high level.");
  const [steps, setSteps] = useState<Step[]>([]);
  const [finalAnswer, setFinalAnswer] = useState("");
  const [diff, setDiff] = useState<string | null>(null);
  const [lastFinal, setLastFinal] = useState("");

  const runDemo = () => {
    const plan = mockPlan(task);
    setSteps(plan);
    const nextFinal = "This is a concise final answer tailored to your task (demo mode, no hidden chain-of-thought).";
    setFinalAnswer(nextFinal);

    // compute “diff” like desktop, but inline
    if (lastFinal && lastFinal.trim() !== nextFinal.trim()) {
      setDiff(simpleDiff(lastFinal, nextFinal));
    } else {
      setDiff(null);
    }
    setLastFinal(nextFinal);
  };

  const scores = useMemo(
    () => scoreProxyAttention(task, steps.map(s => s.note), finalAnswer),
    [task, steps, finalAnswer]
  );

  return (
    <div className="app">
      <div className="header">
        <div className="brand">Ignition — Visual AI Debugger (Web Demo)</div>
        <span className="tag">Mock mode • React + Vite</span>
      </div>

      <div className="panel">
        <label className="subtle">Query / Task</label>
        <textarea
          rows={3}
          placeholder="Ask anything…"
          value={task}
          onChange={(e) => setTask(e.target.value)}
        />
        <div style={{ display: "flex", gap: 12, marginTop: 10 }}>
          <button className="primary" onClick={runDemo}>Run ▶</button>
          <span className="subtle">No GPU or model required.</span>
        </div>
      </div>

      <div className="row">
        <div className="panel">
          <div style={{display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:8}}>
            <h3 style={{margin:0, fontSize:16}}>Plan</h3>
            <span className="badge">{steps.length ? "Generated" : "—"}</span>
          </div>
          <div style={{ display: "grid", gap: 10 }}>
            {steps.map((s) => (
              <div key={s.id} className="step">
                <h4>{s.id}. {s.title}</h4>
                <p>{s.note}</p>
              </div>
            ))}
            {!steps.length && <div className="subtle">Run the demo to see steps.</div>}
          </div>
        </div>

        <div className="panel">
          <div style={{display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:8}}>
            <h3 style={{margin:0, fontSize:16}}>Final Answer</h3>
            <span className="badge">Demo Output</span>
          </div>

          {/* The answer block */}
          <div className="final">{finalAnswer || "—"}</div>

          {/* Inline DIFF (replaces the desktop popup). Appears directly above the heatmap. */}
          {diff && (
            <div style={{marginTop:12}}>
              <div style={{fontWeight:600, marginBottom:6}}>Answer changed (diff)</div>
              <DiffBlock diff={diff} />
            </div>
          )}

          {/* Why Heatmap (proxy) */}
          <div className="heatmap">
            <div style={{fontWeight:600, marginBottom:6}}>Why Heatmap (proxy)</div>
            <HighlightedQuery query={task} scores={scores} />
          </div>
        </div>
      </div>
    </div>
  );
}

/** Minimal inline diff (word-level). */
function simpleDiff(a: string, b: string): string {
  const A = a.split(/\s+/);
  const B = b.split(/\s+/);
  const out: string[] = [];

  const len = Math.max(A.length, B.length);
  for (let i = 0; i < len; i++) {
    const wA = A[i], wB = B[i];
    if (wA === wB) {
      if (wB !== undefined) out.push(wB);
    } else {
      if (wA !== undefined) out.push(`[-${wA}-]`);
      if (wB !== undefined) out.push(`[+${wB}+]`);
    }
  }
  return out.join(" ");
}

function DiffBlock({ diff }: { diff: string }) {
  // Style [-removed-] and [+added+]
  const html = diff
    .replace(/\[\-(.+?)\-\]/g, (_m, p1) => `<span style="background:#3a0a0a;border:1px solid #5b1b1b;padding:1px 3px;border-radius:6px;color:#ffd6d6;text-decoration:line-through;">${escapeHtml(p1)}</span>`)
    .replace(/\[\+(.+?)\+\]/g, (_m, p1) => `<span style="background:#0c2b12;border:1px solid #1f5b2a;padding:1px 3px;border-radius:6px;color:#d6ffde;">${escapeHtml(p1)}</span>`);

  return <div className="subtle" dangerouslySetInnerHTML={{ __html: html }} />;
}

function escapeHtml(s: string) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function HighlightedQuery({ query, scores }: { query: string; scores: Record<string, number> }) {
  const parts = query.match(/\S+|\s+/g) ?? [];
  return (
    <div style={{ lineHeight: 1.9 }}>
      {parts.map((w, i) => {
        const base = w.replace(/\W+$/g, "");
        const key = base.toLowerCase();
        const s = scores[key] ?? 0;
        if (/\s+/.test(w) || !base) return <span key={i}>{w}</span>;
        const alpha = 0.16 + 0.6 * s; // 0.16–0.76
        return (
          <span
            key={i}
            style={{
              background: `rgba(64,132,247,${alpha.toFixed(2)})`,
              padding: "2px 4px",
              borderRadius: 6,
              marginRight: 2,
              display: "inline-block"
            }}
          >
            {w}
          </span>
        );
      })}
    </div>
  );
}

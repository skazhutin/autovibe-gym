/** Shared run-related UI bits used across Dashboard / Runs / Compare. */
import { useNavigate } from "react-router-dom";
import type { Run } from "../lib/api";
import { MODE_SHORT, formatScore } from "../lib/format";
import { StatusBadge, Tag } from "./ui";

export function ModeTag({ mode }: { mode: Run["mode"] }) {
  return <Tag tone={mode === "gym" || mode === "batch" ? "accent" : "neutral"}>{MODE_SHORT[mode]}</Tag>;
}

export function ScoreCell({ run }: { run: Run }) {
  if (run.score === null || run.score === undefined) {
    return <span className="score-cell score-null">—</span>;
  }
  return (
    <span className="score-cell">
      {formatScore(run.score, run.metric)}
      {run.metric && <span className="metric-suffix">{run.metric}</span>}
    </span>
  );
}

export function RunRow({ run }: { run: Run }) {
  const nav = useNavigate();
  return (
    <tr className="clickable" onClick={() => nav(`/runs/${run.id}`)}>
      <td className="mono faint">{run.shortId}</td>
      <td className="mono">{run.model}</td>
      <td><ModeTag mode={run.mode} /></td>
      <td>{run.dataset}</td>
      <td><ScoreCell run={run} /></td>
      <td><StatusBadge status={run.status} /></td>
    </tr>
  );
}

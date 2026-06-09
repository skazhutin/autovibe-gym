import type { Task } from "../../lib/api";
import { formatDateOnly } from "../../lib/date";
import { Card, Tag } from "../ui";
import { STATUS_TONE, STATUS_LABEL, TASK_LABEL, statusOf, sourceText } from "../../lib/taskUtils";

export function splitTag(ok?: boolean, label?: string) {
  return <Tag tone={ok ? "green" : "neutral"}>{label}</Tag>;
}

export function TaskCard({
  d,
  dateFormat,
  onOpen,
  selecting,
  isSelected,
  onToggle,
}: {
  d: Task;
  dateFormat: "mdy" | "dmy";
  onOpen: () => void;
  selecting: boolean;
  isSelected: boolean;
  onToggle: () => void;
}) {
  const status = statusOf(d);
  const taskLabel = TASK_LABEL[d.taskType ?? d.task] ?? d.task;
  return (
    <Card
      className={`ds-card task-card-rich${selecting && isSelected ? " row-selected" : ""}`}
      hover
      onClick={() => (selecting ? onToggle() : onOpen())}
    >
      <div className="spread">
        <div style={{ minWidth: 0 }}>
          <div className="ds-title">{d.name}</div>
        </div>
        <Tag tone={STATUS_TONE[status]}>{STATUS_LABEL[status]}</Tag>
      </div>
      {d.desc && <div className="muted clamp-2">{d.desc}</div>}
      <div className="run-meta-line" style={{ margin: 0 }}>
        <Tag
          tone={d.taskType === "regression" ? "blue" : d.taskType === "classification" ? "accent" : "neutral"}
        >
          {taskLabel}
        </Tag>
        {(d.tags ?? []).slice(0, 3).map((tag) => (
          <Tag key={tag} tone="neutral">{tag}</Tag>
        ))}
        {d.warningsCount ? <Tag tone="red">{d.warningsCount} warnings</Tag> : null}
      </div>
      <div className="ds-stats rich">
        <div className="ds-stat"><span className="k">Rows</span><span className="v">{d.rows ? d.rows.toLocaleString() : "-"}</span></div>
        <div className="ds-stat"><span className="k">Features</span><span className="v">{d.cols || "-"}</span></div>
        <div className="ds-stat"><span className="k">Target</span><span className="v">{d.target}</span></div>
        <div className="ds-stat"><span className="k">Metric</span><span className="v">{d.metric}</span></div>
        <div className="ds-stat"><span className="k">Source</span><span className="v">{sourceText(d)}</span></div>
        {d.createdAt && (
          <div className="ds-stat">
            <span className="k">Created</span>
            <span className="v">{formatDateOnly(d.createdAt, dateFormat)}</span>
          </div>
        )}
      </div>
      <div className="split-pills">
        {splitTag(d.hasTrain, "Train")}
        {splitTag(d.hasVal, "Val")}
        {splitTag(d.hasTest, "Test")}
      </div>
    </Card>
  );
}

import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Icon } from "./Icon";
import type { RunStatus } from "../lib/api";
import { STATUS_LABELS, formatDuration } from "../lib/format";

/* Ticks every second while a run is in progress; otherwise shows final dur. */
export function LiveDuration({ startedMs, running, dur }: { startedMs?: number; running: boolean; dur: number | null | undefined }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [running]);
  const sec = running && startedMs ? Math.max(0, (now - startedMs) / 1000) : dur;
  return <>{formatDuration(sec)}</>;
}

/* ---------------- Button ---------------- */
type BtnVariant = "primary" | "secondary" | "ghost" | "danger";
export function Button({
  children,
  variant = "secondary",
  size = "md",
  icon,
  block,
  ...rest
}: {
  children?: ReactNode;
  variant?: BtnVariant;
  size?: "sm" | "md" | "lg";
  icon?: string;
  block?: boolean;
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...rest}
      className={`btn btn-${variant} btn-${size}${block ? " btn-block" : ""}${
        rest.className ? " " + rest.className : ""
      }`}
    >
      {icon && <Icon name={icon} size={size === "lg" ? 19 : 16} />}
      {children}
    </button>
  );
}

/* ---------------- Card ---------------- */
export function Card({
  children,
  className,
  style,
  onClick,
  hover,
}: {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
  onClick?: () => void;
  hover?: boolean;
}) {
  return (
    <div
      className={`card${hover ? " card-hover" : ""}${className ? " " + className : ""}`}
      style={style}
      onClick={onClick}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      {children}
    </div>
  );
}

/* ---------------- Status dot + badge ---------------- */
export function Dot({ tone, pulse }: { tone: "green" | "red" | "gray" | "accent"; pulse?: boolean }) {
  return <span className={`dot dot-${tone}${pulse ? " dot-pulse" : ""}`} />;
}

const STATUS_TONE: Record<RunStatus, "green" | "red" | "gray" | "accent"> = {
  success: "green",
  failed: "red",
  null: "gray",
  running: "accent",
};

export function StatusBadge({ status }: { status: RunStatus }) {
  return (
    <span className={`badge badge-${status}`}>
      <Dot tone={STATUS_TONE[status]} pulse={status === "running"} />
      {STATUS_LABELS[status]}
    </span>
  );
}

/* ---------------- Tag / chip ---------------- */
export function Tag({
  children,
  tone = "neutral",
  mono,
}: {
  children: ReactNode;
  tone?: "neutral" | "accent" | "green" | "red" | "blue" | "dark";
  mono?: boolean;
}) {
  return <span className={`tag tag-${tone}${mono ? " mono" : ""}`}>{children}</span>;
}

/* ---------------- Progress ring ---------------- */
export function ProgressRing({
  value,
  max,
  size = 46,
  stroke = 5,
  label,
  tone = "accent",
}: {
  value: number;
  max: number;
  size?: number;
  stroke?: number;
  label?: ReactNode;
  tone?: "accent" | "green";
}) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const pct = max > 0 ? Math.min(value / max, 1) : 0;
  return (
    <div className="ring" style={{ width: size, height: size }}>
      <svg width={size} height={size}>
        <circle cx={size / 2} cy={size / 2} r={r} stroke="var(--border)" strokeWidth={stroke} fill="none" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke={tone === "green" ? "var(--green)" : "var(--accent)"}
          strokeWidth={stroke}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - pct)}
          style={{ transition: "stroke-dashoffset .6s cubic-bezier(.4,0,.2,1)", transform: "rotate(-90deg)", transformOrigin: "center" }}
        />
      </svg>
      <span className="ring-label">{label ?? `${value}/${max}`}</span>
    </div>
  );
}

/* ---------------- Progress bar ---------------- */
export function ProgressBar({ pct, animated }: { pct: number; animated?: boolean }) {
  return (
    <div className="pbar">
      <div className="pbar-fill" style={{ width: `${Math.min(Math.max(pct, 0), 100)}%` }}>
        {animated && <span className="pbar-shine" />}
      </div>
    </div>
  );
}

/* ---------------- Tabs ---------------- */
export function Tabs({
  tabs,
  active,
  onChange,
}: {
  tabs: { id: string; label: string; icon?: string; count?: number }[];
  active: string;
  onChange: (id: string) => void;
}) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          aria-selected={active === t.id}
          className={`tab${active === t.id ? " tab-active" : ""}`}
          onClick={() => onChange(t.id)}
        >
          {t.icon && <Icon name={t.icon} size={16} />}
          {t.label}
          {t.count !== undefined && <span className="tab-count">{t.count}</span>}
        </button>
      ))}
    </div>
  );
}

/* ---------------- Skeleton ---------------- */
export function Skeleton({ w, h = 14, style }: { w?: number | string; h?: number; style?: CSSProperties }) {
  return <span className="skeleton" style={{ width: w ?? "100%", height: h, ...style }} />;
}

/* ---------------- Empty state ---------------- */
export function EmptyState({
  icon = "layers",
  title,
  text,
  action,
}: {
  icon?: string;
  title: string;
  text?: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <div className="empty-icon">
        <Icon name={icon} size={26} />
      </div>
      <div className="empty-title">{title}</div>
      {text && <div className="empty-text">{text}</div>}
      {action && <div style={{ marginTop: 14 }}>{action}</div>}
    </div>
  );
}

/* ---------------- Spinner ---------------- */
export function Spinner({ size = 16 }: { size?: number }) {
  return <span className="spinner" style={{ width: size, height: size }} />;
}

/* ---------------- Modal ---------------- */
export function Modal({
  title,
  onClose,
  children,
  footer,
  width = 520,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
  width?: number;
}) {
  return createPortal(
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" style={{ width }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>{title}</h3>
          <button className="icon-btn" onClick={onClose} aria-label="Закрыть">
            <Icon name="x" size={18} />
          </button>
        </div>
        <div className="modal-body">{children}</div>
        {footer && <div className="modal-foot">{footer}</div>}
      </div>
    </div>,
    document.body
  );
}

/* ---------------- SelectDropdown ---------------- */
export function SelectDropdown<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const current = options.find((o) => o.value === value);

  useEffect(() => {
    if (!open) return;
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  return (
    <div className="select-dropdown" ref={ref}>
      <button
        className="select-dropdown-trigger"
        onClick={() => setOpen((v) => !v)}
        type="button"
      >
        <span>{current?.label ?? value}</span>
        <Icon name="chevronDown" size={14} strokeWidth={2.2} />
      </button>
      {open && (
        <div className="select-dropdown-menu">
          {options.map((o) => (
            <button
              key={o.value}
              className={`select-dropdown-item${o.value === value ? " selected" : ""}`}
              onClick={() => { onChange(o.value); setOpen(false); }}
              type="button"
            >
              {o.value === value && <Icon name="check" size={13} strokeWidth={2.5} />}
              {o.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/* ---------------- Form field ---------------- */
export function Field({ label, hint, children, required }: { label: ReactNode; hint?: string; children: ReactNode; required?: boolean }) {
  return (
    <label className="field">
      <span className="field-label">
        {label}
        {required && <span className="required-star" aria-hidden="true">*</span>}
      </span>
      {hint && <span className="field-hint">{hint}</span>}
      {children}
    </label>
  );
}

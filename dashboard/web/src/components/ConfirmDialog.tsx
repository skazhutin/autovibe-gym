import { useEffect } from "react";
import { createPortal } from "react-dom";
import { Button, Spinner } from "./ui";

export function ConfirmDialog({
  title,
  description,
  confirmLabel = "Подтвердить",
  confirmVariant = "primary",
  cancelLabel = "Отменить",
  busy = false,
  onConfirm,
  onCancel,
}: {
  title: string;
  description?: string;
  confirmLabel?: string;
  confirmVariant?: "primary" | "danger";
  cancelLabel?: string;
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) { if (e.key === "Escape") onCancel(); }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onCancel]);

  return createPortal(
    <div className="modal-backdrop" onClick={onCancel}>
      <div className="modal-box" onClick={e => e.stopPropagation()}>
        <h3 className="modal-title">{title}</h3>
        {description && <p className="modal-desc">{description}</p>}
        <div className="modal-actions">
          <Button variant="ghost" onClick={onCancel} disabled={busy}>{cancelLabel}</Button>
          <Button variant={confirmVariant} onClick={onConfirm} disabled={busy}>
            {busy ? <Spinner size={14} /> : confirmLabel}
          </Button>
        </div>
      </div>
    </div>,
    document.body
  );
}

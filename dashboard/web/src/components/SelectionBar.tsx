import { createPortal } from "react-dom";
import { Button, Spinner } from "./ui";

type Noun = "прогон" | "задача" | "модель";

function pluralize(n: number, noun: Noun): string {
  const mod10 = n % 10, mod100 = n % 100;
  const many = mod100 >= 11 && mod100 <= 14;
  if (noun === "прогон") return many || mod10 === 0 || mod10 >= 5 ? `${n} прогонов` : mod10 === 1 ? `${n} прогон` : `${n} прогона`;
  if (noun === "задача") return many || mod10 === 0 || mod10 >= 5 ? `${n} задач` : mod10 === 1 ? `${n} задача` : `${n} задачи`;
  return many || mod10 === 0 || mod10 >= 5 ? `${n} моделей` : mod10 === 1 ? `${n} модель` : `${n} модели`;
}

export function SelectionBar({
  count,
  noun,
  actionLabel,
  actionIcon,
  busy = false,
  onAction,
  onCancel,
}: {
  count: number;
  noun: Noun;
  actionLabel: string;
  actionIcon?: string;
  busy?: boolean;
  onAction: () => void;
  onCancel: () => void;
}) {
  return createPortal(
    <div className="selection-bar">
      <span className="selection-bar-label">Выбрано: {pluralize(count, noun)}</span>
      <Button variant="primary" icon={actionIcon} onClick={onAction} disabled={busy}>
        {busy ? <Spinner size={14} /> : actionLabel}
      </Button>
      <Button variant="ghost" onClick={onCancel} disabled={busy}>Отменить</Button>
    </div>,
    document.body
  );
}

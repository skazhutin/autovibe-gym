export type DateFormat = "mdy" | "dmy";

export function formatDateOnly(value: string | number | Date, format: DateFormat = "mdy"): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  const day = String(date.getDate()).padStart(2, "0");
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const year = String(date.getFullYear());
  return format === "dmy" ? `${day}.${month}.${year}` : `${month}/${day}/${year}`;
}

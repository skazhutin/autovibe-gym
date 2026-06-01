/**
 * App shell placeholder. The real sidebar + header + routed screens are built
 * in the next commit (frontend foundation). This keeps the skeleton runnable.
 */
export default function App() {
  return (
    <div
      style={{
        display: "flex",
        height: "100%",
        alignItems: "center",
        justifyContent: "center",
        flexDirection: "column",
        gap: 12,
      }}
    >
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontWeight: 800,
          fontSize: 28,
          letterSpacing: "-0.02em",
        }}
      >
        AutoVibe <span style={{ color: "var(--accent-ink)" }}>Gym</span>
      </div>
      <div style={{ color: "var(--text-dim)" }}>
        Скелет дашборда. Экраны и оболочка — в следующих коммитах.
      </div>
    </div>
  );
}

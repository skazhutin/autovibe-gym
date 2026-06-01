/** Minimal Python syntax highlighter (token coloring per design tokens). */
import { Fragment, type ReactNode } from "react";

const KEYWORDS = new Set([
  "and","as","assert","async","await","break","class","continue","def","del","elif","else",
  "except","finally","for","from","global","if","import","in","is","lambda","nonlocal","not",
  "or","pass","raise","return","try","while","with","yield","None","True","False","self",
]);

type Tok = { t: string; cls?: string };

function tokenizeLine(line: string): Tok[] {
  const toks: Tok[] = [];
  let i = 0;
  const n = line.length;
  while (i < n) {
    const ch = line[i];
    if (ch === "#") {
      toks.push({ t: line.slice(i), cls: "tok-comment" });
      break;
    }
    if (ch === '"' || ch === "'") {
      let j = i + 1;
      while (j < n && line[j] !== ch) {
        if (line[j] === "\\") j++;
        j++;
      }
      toks.push({ t: line.slice(i, Math.min(j + 1, n)), cls: "tok-string" });
      i = j + 1;
      continue;
    }
    if (/[0-9]/.test(ch)) {
      let j = i;
      while (j < n && /[0-9._eE+-]/.test(line[j])) j++;
      toks.push({ t: line.slice(i, j), cls: "tok-number" });
      i = j;
      continue;
    }
    if (/[A-Za-z_]/.test(ch)) {
      let j = i;
      while (j < n && /[A-Za-z0-9_]/.test(line[j])) j++;
      const word = line.slice(i, j);
      const isCall = line[j] === "(";
      toks.push({ t: word, cls: KEYWORDS.has(word) ? "tok-keyword" : isCall ? "tok-function" : undefined });
      i = j;
      continue;
    }
    if (/[+\-*/%=<>!&|^~@.,:]/.test(ch)) {
      toks.push({ t: ch, cls: "tok-operator" });
      i++;
      continue;
    }
    toks.push({ t: ch });
    i++;
  }
  return toks;
}

export function CodeBlock({ code, className }: { code: string; className?: string }) {
  const lines = (code ?? "").replace(/\n$/, "").split("\n");
  return (
    <pre className={`code${className ? " " + className : ""}`}>
      <code>
        {lines.map((line, li) => (
          <Fragment key={li}>
            {tokenizeLine(line).map((tok, ti) =>
              tok.cls ? (
                <span key={ti} className={tok.cls}>
                  {tok.t}
                </span>
              ) : (
                (<span key={ti}>{tok.t}</span>) as ReactNode
              )
            )}
            {li < lines.length - 1 ? "\n" : ""}
          </Fragment>
        ))}
      </code>
    </pre>
  );
}

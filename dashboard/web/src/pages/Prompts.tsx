/**
 * /prompts — block-aware system prompt editor.
 *
 * Design notes (see commit message and CLAUDE.md/docs/STATUS.md):
 *  - The `default` preset is synthesized from gym.prompts on the backend
 *    every time it is fetched. It is read-only here; the only action is
 *    "Сделать копию" which seeds a new editable preset.
 *  - Each block carries a tier. Locked = readonly + lock icon. Trusted =
 *    editable but the first edit shows a confirm dialog. Editable = free.
 *  - Saving an empty `blocks` map is allowed (a named copy of default).
 *  - Top warning banner is permanent — the prompt drives the agent's
 *    contract with the environment and the dashboard wants this front and
 *    centre, not buried in a tooltip.
 *  - Warnings returned by the backend's sanity check are shown inline.
 *    They are NEVER fatal — saving still works — but flagging them keeps
 *    the operator honest.
 */
import { useEffect, useMemo, useState } from "react";
import {
  api,
  type PromptBlockTier,
  type PromptPresetDetail,
  type PromptPresetSummary,
} from "../lib/api";
import { useAsync } from "../lib/hooks";
import {
  Button,
  Card,
  EmptyState,
  Field,
  Modal,
  Skeleton,
  Spinner,
  Tag,
} from "../components/ui";
import { Icon } from "../components/Icon";

// Human-readable labels for the named blocks. Keep these aligned with
// BLOCK_ORDER in gym/prompts.py — if the backend grows a new block the
// page still renders it (unknown key just falls back to the id).
const BLOCK_LABELS: Record<string, string> = {
  header: "Header",
  kernel_vars: "Kernel variables",
  libraries: "Installed libraries",
  critical_rules: "Critical rules",
  tools_hint: "Tool preference hints",
  failure_patterns: "Common failure patterns",
  finalize: "Finalize-early reminder",
  submit_recovery: "Submit-failure recovery",
  clean_run_tips: "Clean-run robustness tips",
  thoughts_on: "Thoughts mode (enabled)",
  thoughts_off: "Thoughts mode (disabled)",
};

const BLOCK_HINTS: Record<string, string> = {
  kernel_vars:
    "Locked — these names are bootstrapped into the kernel by gym/notebook_env.py. Changing them here would lie to the agent about what's available.",
  critical_rules:
    "Removing entries here is a common way to silently break the agent. Keep the restart_and_run_all / validate / submit / raw-rows pipeline guarantees intact.",
  submit_recovery:
    "These are the recovery instructions shown after a hidden-test failure. Removing the pipeline-robustness bullet causes most submit retries to fail.",
  thoughts_on:
    "Active only when «Thoughts mode» is enabled on a run. The first action must be a `think` with stage `planning`.",
  thoughts_off:
    "Active only when «Thoughts mode» is OFF. Without this block the agent may emit thoughts fields the parser rejects.",
};

const TIER_LABEL: Record<PromptBlockTier, string> = {
  locked: "Locked",
  trusted: "Trusted",
  editable: "Editable",
};

const TIER_TONE: Record<PromptBlockTier, "red" | "accent" | "neutral"> = {
  locked: "red",
  trusted: "accent",
  editable: "neutral",
};

interface NewPresetModalProps {
  onClose: () => void;
  onDone: (id: string) => void;
}

function NewPresetModal({ onClose, onDone }: NewPresetModalProps) {
  const [id, setId] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const idValid = /^[a-z0-9][a-z0-9_-]{0,39}$/.test(id);

  async function save() {
    setBusy(true);
    setError(null);
    try {
      // Empty blocks → preset starts as a named copy of default.
      await api.savePrompt({ id, name, blocks: {} });
      onDone(id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Не удалось создать пресет");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal
      title="Новый пресет промпта"
      width={460}
      onClose={onClose}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>Отмена</Button>
          <Button
            variant="primary"
            onClick={save}
            disabled={busy || !idValid || !name.trim()}
          >
            {busy ? <Spinner /> : "Создать"}
          </Button>
        </>
      }
    >
      <div className="stack" style={{ gap: 14 }}>
        <Field
          label="ID"
          hint="[a-z0-9_-], до 40 символов. Используется в URL и MLflow-тегах."
        >
          <input
            className="input mono"
            value={id}
            onChange={(e) => setId(e.target.value.toLowerCase())}
            placeholder="minimal"
          />
        </Field>
        <Field label="Название" hint="Как пресет будет отображаться в /new">
          <input
            className="input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Минимальный промпт"
          />
        </Field>
        {id && !idValid && (
          <div className="prompt-error">
            ID должен начинаться с буквы/цифры и содержать только a-z, 0-9, _, -
          </div>
        )}
        {error && <div className="prompt-error">{error}</div>}
      </div>
    </Modal>
  );
}

interface BlockEditorProps {
  name: string;
  tier: PromptBlockTier;
  defaultValue: string;
  currentValue: string;
  overridden: boolean;
  isLocked: boolean;
  readOnly: boolean;
  onChange: (value: string) => void;
  onReset: () => void;
}

function BlockEditor({
  name,
  tier,
  defaultValue,
  currentValue,
  overridden,
  isLocked,
  readOnly,
  onChange,
  onReset,
}: BlockEditorProps) {
  const [warned, setWarned] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const label = BLOCK_LABELS[name] ?? name;
  const hint = BLOCK_HINTS[name];
  const rowCount = Math.max(3, Math.min(24, (currentValue.match(/\n/g) || []).length + 1));

  function handleChange(next: string) {
    if (isLocked || readOnly) return;
    if (tier === "trusted" && !warned && next !== defaultValue) {
      // Defer the actual edit until the user confirms; the confirm modal
      // re-applies it on accept. This stops a single accidental keystroke
      // from silently mutating a critical block.
      setShowConfirm(true);
      return;
    }
    onChange(next);
  }

  return (
    <Card className="prompt-block">
      <div className="prompt-block-head">
        <span className="prompt-block-title mono">{name}</span>
        <span className="prompt-block-label">{label}</span>
        <span style={{ flex: 1 }} />
        {overridden && !isLocked && (
          <Tag tone="accent" mono>modified</Tag>
        )}
        <Tag tone={TIER_TONE[tier]} mono>
          {isLocked && <Icon name="lock" size={11} />}
          {TIER_LABEL[tier]}
        </Tag>
      </div>
      {hint && <div className="prompt-block-hint">{hint}</div>}
      <textarea
        className={`input mono prompt-textarea${isLocked ? " locked" : ""}`}
        value={currentValue}
        readOnly={isLocked || readOnly}
        onChange={(e) => handleChange(e.target.value)}
        rows={rowCount}
        spellCheck={false}
      />
      {overridden && !isLocked && !readOnly && (
        <div className="prompt-block-actions">
          <Button variant="ghost" size="sm" onClick={onReset}>
            Сбросить блок к default
          </Button>
        </div>
      )}
      {showConfirm && (
        <Modal
          title="Изменить trusted-блок?"
          width={460}
          onClose={() => setShowConfirm(false)}
          footer={
            <>
              <Button variant="ghost" onClick={() => setShowConfirm(false)}>Отмена</Button>
              <Button
                variant="danger"
                onClick={() => {
                  setWarned(true);
                  setShowConfirm(false);
                  // We don't know the *new* value here because handleChange
                  // returned early. The user simply taps the textarea again
                  // after confirming. This avoids accidentally applying a
                  // half-typed value. Trade-off: one extra keystroke after
                  // confirm; documented in the design.
                }}
              >
                Я понимаю риски
              </Button>
            </>
          }
        >
          <div style={{ fontSize: 14, lineHeight: 1.55 }}>
            Этот блок (<b>{label}</b>) описывает контракт между агентом и средой.
            Удаление пунктов про <span className="mono">restart_and_run_all</span>,{" "}
            <span className="mono">validate/submit</span>,{" "}
            <span className="mono">model_var</span> или работу на сырых строках
            ломает большинство прогонов.
            <br /><br />
            Подтвердите, что хотите редактировать, затем продолжите ввод в поле.
            При сохранении вы увидите предупреждения, если ключевые фразы пропали.
          </div>
        </Modal>
      )}
    </Card>
  );
}

export default function Prompts() {
  const { data: list, loading: listLoading, reload: reloadList } = useAsync(
    () => api.listPrompts(),
    [],
  );
  const [selectedId, setSelectedId] = useState<string>("default");
  const [detail, setDetail] = useState<PromptPresetDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  // Working copy of overrides — keyed by block name. Empty/null means "use default".
  const [draftBlocks, setDraftBlocks] = useState<Record<string, string>>({});
  const [draftName, setDraftName] = useState("");
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  useEffect(() => {
    if (!list || list.items.length === 0) return;
    if (!list.items.some((p) => p.id === selectedId)) {
      setSelectedId(list.default_id);
    }
  }, [list, selectedId]);

  useEffect(() => {
    let cancelled = false;
    setDetailLoading(true);
    setError(null);
    api
      .getPrompt(selectedId)
      .then((d) => {
        if (cancelled) return;
        setDetail(d);
        // Reset draft to whatever overrides the preset already has.
        setDraftBlocks({ ...d.block_overrides });
        setDraftName(d.name);
        setDirty(false);
      })
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : "Не удалось загрузить пресет"))
      .finally(() => !cancelled && setDetailLoading(false));
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  const isDefault = detail?.is_default === true;
  const isReadOnly = isDefault;

  const effectiveBlocks: Record<string, string> = useMemo(() => {
    if (!detail) return {};
    const out: Record<string, string> = {};
    for (const name of detail.block_order) {
      // For default preset there is no override map — show defaults.
      out[name] = name in draftBlocks ? draftBlocks[name] : detail.blocks[name];
    }
    return out;
  }, [detail, draftBlocks]);

  function setBlock(name: string, value: string) {
    if (!detail) return;
    setDraftBlocks((prev) => {
      const next = { ...prev };
      // If user typed exactly the default, clear the override.
      if (value === detail.blocks[name] && !(name in detail.block_overrides)) {
        delete next[name];
      } else {
        next[name] = value;
      }
      return next;
    });
    setDirty(true);
  }

  function resetBlock(name: string) {
    if (!detail) return;
    setDraftBlocks((prev) => {
      const next = { ...prev };
      delete next[name];
      return next;
    });
    setDirty(true);
  }

  async function save() {
    if (!detail) return;
    setBusy(true);
    setError(null);
    try {
      const saved = await api.savePrompt({
        id: detail.id,
        name: draftName.trim() || detail.name,
        blocks: draftBlocks,
      });
      setDetail(saved);
      setDraftBlocks({ ...saved.block_overrides });
      setDraftName(saved.name);
      setDirty(false);
      reloadList();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Не удалось сохранить пресет");
    } finally {
      setBusy(false);
    }
  }

  async function resetAll() {
    if (!detail || isReadOnly) return;
    setDraftBlocks({});
    setDirty(true);
  }

  async function doDelete() {
    if (!detail) return;
    setBusy(true);
    try {
      await api.deletePrompt(detail.id);
      setConfirmDelete(false);
      setSelectedId("default");
      reloadList();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Не удалось удалить пресет");
    } finally {
      setBusy(false);
    }
  }

  async function duplicateDefault() {
    // Triggered from the read-only default view: opens the new-preset modal
    // with no prefill; the new preset starts empty (= a named clone of default)
    // that the user can then customise.
    setAdding(true);
  }

  return (
    <div className="prompts-page">
      <aside className="prompts-aside">
        <div className="prompts-aside-head">
          <h3 style={{ margin: 0, fontSize: 14 }}>Пресеты</h3>
          <Button
            variant="primary"
            size="sm"
            icon="plus"
            onClick={() => setAdding(true)}
          >
            Новый
          </Button>
        </div>
        {listLoading && <Skeleton h={28} />}
        {list && (
          <div className="prompts-list">
            {list.items.map((p) => (
              <PresetRow
                key={p.id}
                preset={p}
                active={p.id === selectedId}
                onSelect={() => setSelectedId(p.id)}
              />
            ))}
          </div>
        )}
        <div className="prompts-warn-banner">
          <Icon name="alert" size={16} />
          <div>
            <b>Осторожно.</b> Системный промпт диктует контракт между агентом и
            средой. Удаление ключевых пунктов (restart_and_run_all, validate /
            submit, model_var, raw-rows pipeline) ломает большинство прогонов.
            Всегда есть кнопка «Сбросить к default».
          </div>
        </div>
      </aside>

      <section className="prompts-main">
        {detailLoading && !detail && <Skeleton h={28} />}
        {!detail && !detailLoading && (
          <EmptyState icon="terminal" title="Выберите пресет слева" />
        )}
        {detail && (
          <>
            <div className="prompts-header">
              <div>
                {isReadOnly ? (
                  <h2 className="prompts-title">{detail.name}</h2>
                ) : (
                  <input
                    className="input prompts-title-input"
                    value={draftName}
                    onChange={(e) => {
                      setDraftName(e.target.value);
                      setDirty(true);
                    }}
                  />
                )}
                <div className="prompts-meta mono">
                  id: {detail.id} · sha {detail.sha256.slice(0, 12)}
                  {isReadOnly && <Tag tone="green" mono>default</Tag>}
                </div>
              </div>
              <div className="row" style={{ gap: 8 }}>
                {isReadOnly ? (
                  <Button variant="primary" icon="plus" onClick={duplicateDefault}>
                    Сделать копию
                  </Button>
                ) : (
                  <>
                    <Button variant="ghost" icon="refresh" onClick={resetAll}>
                      Сбросить всё
                    </Button>
                    <Button
                      variant="danger"
                      icon="trash"
                      onClick={() => setConfirmDelete(true)}
                    >
                      Удалить
                    </Button>
                    <Button
                      variant="primary"
                      onClick={save}
                      disabled={busy || !dirty}
                    >
                      {busy ? <Spinner /> : "Сохранить"}
                    </Button>
                  </>
                )}
              </div>
            </div>

            {detail.warnings.length > 0 && (
              <Card className="prompt-warnings">
                <div className="prompt-warnings-head">
                  <Icon name="alert" size={16} />
                  <b>Проверка содержания</b>
                  <span className="faint" style={{ fontSize: 12 }}>
                    предупреждения, не блокирующие сохранение
                  </span>
                </div>
                <ul>
                  {detail.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              </Card>
            )}

            {error && <div className="prompt-error">{error}</div>}

            {detail.block_order.map((name) => (
              <BlockEditor
                key={name}
                name={name}
                tier={detail.block_tiers[name] ?? "editable"}
                defaultValue={detail.blocks[name] /* default text from backend */}
                currentValue={effectiveBlocks[name] ?? ""}
                overridden={name in draftBlocks}
                isLocked={detail.locked_blocks.includes(name)}
                readOnly={isReadOnly}
                onChange={(v) => setBlock(name, v)}
                onReset={() => resetBlock(name)}
              />
            ))}

            <Card className="prompt-block">
              <div className="prompt-block-head">
                <span className="prompt-block-title mono">thoughts_on</span>
                <span className="prompt-block-label">{BLOCK_LABELS.thoughts_on}</span>
                <span style={{ flex: 1 }} />
                <Tag tone="accent" mono>Trusted</Tag>
              </div>
              <div className="prompt-block-hint">{BLOCK_HINTS.thoughts_on}</div>
              <textarea
                className="input mono prompt-textarea"
                value={detail.thoughts_on}
                readOnly
                rows={6}
                spellCheck={false}
              />
              <div className="prompt-block-actions faint" style={{ fontSize: 12 }}>
                Редактирование thoughts-блоков в первой итерации недоступно — это часть протокола (think action).
              </div>
            </Card>
            <Card className="prompt-block">
              <div className="prompt-block-head">
                <span className="prompt-block-title mono">thoughts_off</span>
                <span className="prompt-block-label">{BLOCK_LABELS.thoughts_off}</span>
                <span style={{ flex: 1 }} />
                <Tag tone="accent" mono>Trusted</Tag>
              </div>
              <div className="prompt-block-hint">{BLOCK_HINTS.thoughts_off}</div>
              <textarea
                className="input mono prompt-textarea"
                value={detail.thoughts_off}
                readOnly
                rows={5}
                spellCheck={false}
              />
            </Card>
          </>
        )}
      </section>

      {adding && (
        <NewPresetModal
          onClose={() => setAdding(false)}
          onDone={(id) => {
            setAdding(false);
            setSelectedId(id);
            reloadList();
          }}
        />
      )}

      {confirmDelete && detail && (
        <Modal
          title={`Удалить пресет «${detail.name}»?`}
          width={420}
          onClose={() => setConfirmDelete(false)}
          footer={
            <>
              <Button variant="ghost" onClick={() => setConfirmDelete(false)}>Отмена</Button>
              <Button variant="danger" onClick={doDelete} disabled={busy}>
                {busy ? <Spinner /> : "Удалить"}
              </Button>
            </>
          }
        >
          Удаление необратимо. Прогоны, использовавшие этот пресет, продолжат
          ссылаться на его id и sha в MLflow, но просмотреть содержимое из
          UI будет невозможно.
        </Modal>
      )}
    </div>
  );
}

function PresetRow({
  preset,
  active,
  onSelect,
}: {
  preset: PromptPresetSummary;
  active: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      className={`prompt-row${active ? " active" : ""}`}
      onClick={onSelect}
      type="button"
    >
      <div className="prompt-row-name">{preset.name}</div>
      <div className="prompt-row-meta">
        <span className="mono">{preset.id}</span>
        {preset.is_default ? (
          <Tag tone="green" mono>default</Tag>
        ) : preset.block_override_count > 0 ? (
          <Tag tone="accent" mono>{preset.block_override_count} overrides</Tag>
        ) : (
          <Tag tone="neutral" mono>clone</Tag>
        )}
      </div>
    </button>
  );
}

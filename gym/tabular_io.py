from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable
import zipfile

import pandas as pd


PREVIEW_FULL_READ_LIMIT_BYTES = 32 * 1024 * 1024


class TabularIOError(ValueError):
    """Raised when a file cannot be parsed as a supported tabular dataset."""


@dataclass(frozen=True)
class TabularLoadResult:
    dataframe: pd.DataFrame
    format: str
    warnings: list[str]
    exact_rows_known: bool


@dataclass(frozen=True)
class ReaderSpec:
    format_name: str
    supports_partial_preview: bool
    dependency_hint: str | None
    loader: Callable[[Path, dict[str, Any], int | None], pd.DataFrame]


def _csv_loader(path: Path, options: dict[str, Any], nrows: int | None) -> pd.DataFrame:
    kwargs = dict(options)
    if nrows is not None:
        kwargs.setdefault("nrows", nrows)
    return pd.read_csv(path, **kwargs)


def _tsv_loader(path: Path, options: dict[str, Any], nrows: int | None) -> pd.DataFrame:
    kwargs = dict(options)
    kwargs.setdefault("sep", "\t")
    if nrows is not None:
        kwargs.setdefault("nrows", nrows)
    return pd.read_csv(path, **kwargs)


def _txt_loader(path: Path, options: dict[str, Any], nrows: int | None) -> pd.DataFrame:
    kwargs = dict(options)
    if "sep" not in kwargs and "delimiter" not in kwargs:
        kwargs["sep"] = None
        kwargs.setdefault("engine", "python")
    if nrows is not None:
        kwargs.setdefault("nrows", nrows)
    return pd.read_csv(path, **kwargs)


def _excel_loader(path: Path, options: dict[str, Any], nrows: int | None) -> pd.DataFrame:
    kwargs = dict(options)
    if nrows is not None:
        kwargs.setdefault("nrows", nrows)
    return pd.read_excel(path, **kwargs)


def _json_loader(
    path: Path,
    options: dict[str, Any],
    nrows: int | None,
    *,
    default_lines: bool,
) -> pd.DataFrame:
    kwargs = dict(options)
    if default_lines and "lines" not in kwargs:
        kwargs["lines"] = True
    if nrows is not None and kwargs.get("lines"):
        kwargs.setdefault("nrows", nrows)
    df = pd.read_json(path, **kwargs)
    if nrows is not None and not kwargs.get("lines"):
        return df.head(nrows)
    return df


def _parquet_loader(path: Path, options: dict[str, Any], nrows: int | None) -> pd.DataFrame:
    df = pd.read_parquet(path, **dict(options))
    if nrows is not None:
        return df.head(nrows)
    return df


def _feather_loader(path: Path, options: dict[str, Any], nrows: int | None) -> pd.DataFrame:
    df = pd.read_feather(path, **dict(options))
    if nrows is not None:
        return df.head(nrows)
    return df


def _orc_loader(path: Path, options: dict[str, Any], nrows: int | None) -> pd.DataFrame:
    df = pd.read_orc(path, **dict(options))
    if nrows is not None:
        return df.head(nrows)
    return df


READER_REGISTRY: dict[str, ReaderSpec] = {
    "csv": ReaderSpec("csv", True, None, _csv_loader),
    "csv.gz": ReaderSpec("csv.gz", True, None, _csv_loader),
    "tsv": ReaderSpec("tsv", True, None, _tsv_loader),
    "txt": ReaderSpec("txt", True, None, _txt_loader),
    "xlsx": ReaderSpec("xlsx", True, "openpyxl", _excel_loader),
    "xls": ReaderSpec("xls", True, "xlrd", _excel_loader),
    "json": ReaderSpec("json", True, None, lambda path, opts, nrows: _json_loader(path, opts, nrows, default_lines=False)),
    "jsonl": ReaderSpec("jsonl", True, None, lambda path, opts, nrows: _json_loader(path, opts, nrows, default_lines=True)),
    "ndjson": ReaderSpec("ndjson", True, None, lambda path, opts, nrows: _json_loader(path, opts, nrows, default_lines=True)),
    "parquet": ReaderSpec("parquet", False, "pyarrow or fastparquet", _parquet_loader),
    "feather": ReaderSpec("feather", False, "pyarrow", _feather_loader),
    "orc": ReaderSpec("orc", False, "pyarrow", _orc_loader),
}

SUPPORTED_FORMATS = tuple(READER_REGISTRY.keys()) + ("zip",)


def infer_format(path_or_name: str | Path, explicit_format: str | None = None) -> str:
    if explicit_format and explicit_format.strip().lower() not in {"", "auto"}:
        fmt = explicit_format.strip().lower()
        if fmt not in SUPPORTED_FORMATS:
            raise TabularIOError(
                f"Unsupported format '{explicit_format}'. Supported formats: {', '.join(SUPPORTED_FORMATS)}."
            )
        return fmt

    name = str(path_or_name).lower()
    if name.endswith(".csv.gz"):
        return "csv.gz"
    if name.endswith(".jsonl"):
        return "jsonl"
    if name.endswith(".ndjson"):
        return "ndjson"
    for fmt in ("csv", "tsv", "txt", "parquet", "xlsx", "xls", "json", "zip", "feather", "orc"):
        if name.endswith(f".{fmt}"):
            return fmt
    raise TabularIOError(
        f"Could not infer tabular format from '{path_or_name}'. Supported formats: {', '.join(SUPPORTED_FORMATS)}."
    )


def list_archive_members(path: Path) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            member_name = PurePosixPath(info.filename).as_posix()
            if not is_safe_archive_member(member_name):
                continue
            try:
                fmt = infer_format(member_name)
            except TabularIOError:
                continue
            members.append(
                {
                    "member": member_name,
                    "format": fmt,
                    "size": int(info.file_size),
                }
            )
    return members


def is_supported_tabular_path(path_or_name: str | Path) -> bool:
    try:
        infer_format(path_or_name)
    except TabularIOError:
        return False
    return True


def is_safe_archive_member(member_name: str) -> bool:
    if not member_name or member_name.startswith("/"):
        return False
    path = PurePosixPath(member_name)
    return ".." not in path.parts


def load_tabular_dataframe(
    path: Path,
    *,
    format_name: str = "auto",
    read_options: dict[str, Any] | None = None,
    nrows: int | None = None,
    preview_limit_bytes: int = PREVIEW_FULL_READ_LIMIT_BYTES,
) -> TabularLoadResult:
    fmt = infer_format(path.name, format_name)
    if fmt == "zip":
        raise TabularIOError(
            f"'{path.name}' is a ZIP archive. Choose an archive member before loading it as a table."
        )

    if not path.exists():
        raise FileNotFoundError(f"Missing tabular file: {path}")

    spec = READER_REGISTRY[fmt]
    warnings: list[str] = []
    effective_nrows = nrows
    exact_rows_known = True

    if nrows is not None and not spec.supports_partial_preview:
        if path.stat().st_size > preview_limit_bytes:
            raise TabularIOError(
                f"Preview for '{path.name}' requires reading the full {fmt} file, but it exceeds the "
                f"{preview_limit_bytes // (1024 * 1024)} MB preview limit."
            )
        warnings.append(
            f"Preview for '{path.name}' loaded the full {fmt} file because partial reads are not supported."
        )
        effective_nrows = None

    try:
        df = spec.loader(path, dict(read_options or {}), effective_nrows)
    except ImportError as exc:
        hint = spec.dependency_hint or "an optional dependency"
        raise TabularIOError(
            f"Reading '{path.name}' as {fmt} requires {hint}: {exc}"
        ) from exc
    except ModuleNotFoundError as exc:
        hint = spec.dependency_hint or "an optional dependency"
        raise TabularIOError(
            f"Reading '{path.name}' as {fmt} requires {hint}: {exc}"
        ) from exc
    except ValueError as exc:
        message = str(exc)
        if fmt == "xls" and "engine" in message.lower():
            raise TabularIOError(
                f"Reading '{path.name}' as xls requires xlrd or an explicit compatible engine."
            ) from exc
        raise TabularIOError(f"Failed to read '{path.name}' as {fmt}: {exc}") from exc
    except Exception as exc:
        raise TabularIOError(f"Failed to read '{path.name}' as {fmt}: {exc}") from exc

    if df.empty:
        raise TabularIOError(f"Parsed '{path.name}' as {fmt}, but it is empty.")

    if nrows is not None and effective_nrows is not None and len(df) >= nrows:
        exact_rows_known = False
    return TabularLoadResult(
        dataframe=df,
        format=fmt,
        warnings=warnings,
        exact_rows_known=exact_rows_known,
    )

"""Common no-repair submission validation and one-shot hidden evaluation."""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cloudpickle
import numpy as np
import pandas as pd


DEFAULT_PREDICTION_TIMEOUT_SECONDS = 120.0
DOCKER_PREDICTION_WORKER = """\
from pathlib import Path
import cloudpickle
import json
import numpy as np

root = Path('/work')
try:
    model = cloudpickle.loads((root / 'model.pkl').read_bytes())
    features = cloudpickle.loads((root / 'features.pkl').read_bytes())
    values = np.asarray(model.predict(features))
    if values.ndim != 1:
        raise ValueError('Predictions must be a one-dimensional vector.')
    encoded = []
    for value in values.tolist():
        if isinstance(value, np.generic):
            value = value.item()
        if value is not None and not isinstance(value, (str, bool, int, float)):
            raise TypeError(
                'Predictions must contain only JSON scalar values, not '
                + type(value).__name__
            )
        encoded.append(value)
    result = {'ok': True, 'predictions': encoded}
    payload = json.dumps(result, allow_nan=False)
except BaseException as error:
    result = {
        'ok': False,
        'error_type': type(error).__name__,
        'error_message': str(error)[:2000],
    }
    payload = json.dumps(result, allow_nan=False)
(root / 'result.json').write_text(payload, encoding='utf-8')
"""


@dataclass
class SubmissionValidation:
    valid: bool
    exists: bool
    has_predict: bool
    serializable: bool | None = None
    raw_prediction_ok: bool | None = None
    prediction_length_ok: bool | None = None
    prediction_nan_free: bool | None = None
    error_type: str | None = None
    error_message: str | None = None
    predictions: Any = field(default=None, repr=False, compare=False)


def prediction_backend_for_execution(execution_backend: str | None) -> str:
    return (
        "docker"
        if (execution_backend or "").strip().lower() == "docker"
        else "process"
    )


def validate_submission_candidate(
    model: Any,
    raw_features: pd.DataFrame,
    *,
    timeout_seconds: float | None = None,
    prediction_backend: str | None = None,
) -> SubmissionValidation:
    if model is None:
        return SubmissionValidation(
            valid=False,
            exists=False,
            has_predict=False,
            error_type="CandidateMissing",
            error_message="No candidate object was provided.",
        )
    has_predict = callable(getattr(model, "predict", None))
    if not has_predict:
        return SubmissionValidation(
            valid=False,
            exists=True,
            has_predict=False,
            error_type="ModelInterfaceError",
            error_message="Candidate does not expose predict(X).",
        )
    try:
        model_payload = cloudpickle.dumps(model)
    except Exception as error:
        return SubmissionValidation(
            valid=False,
            exists=True,
            has_predict=True,
            serializable=False,
            error_type=type(error).__name__,
            error_message=str(error),
        )
    prediction = _predict_isolated(
        model_payload,
        raw_features,
        timeout_seconds=_prediction_timeout(timeout_seconds),
        prediction_backend=prediction_backend,
    )
    if not prediction["ok"]:
        return SubmissionValidation(
            valid=False,
            exists=True,
            has_predict=True,
            serializable=True,
            raw_prediction_ok=False,
            error_type=prediction["error_type"],
            error_message=prediction["error_message"],
        )
    predictions = prediction["predictions"]
    length_ok = (
        _prediction_length(predictions) == len(raw_features)
        and _predictions_one_dimensional(predictions)
    )
    nan_free = _predictions_nan_free(predictions)
    return SubmissionValidation(
        valid=bool(length_ok and nan_free),
        exists=True,
        has_predict=True,
        serializable=True,
        raw_prediction_ok=True,
        prediction_length_ok=length_ok,
        prediction_nan_free=nan_free,
        error_type=None if length_ok and nan_free else "PredictionContractError",
        error_message=None if length_ok and nan_free else "Predictions must be one non-null value per row.",
        predictions=predictions,
    )


class HiddenEvaluationAlreadyAttempted(RuntimeError):
    pass


class HiddenEvaluationGate:
    def __init__(
        self,
        *,
        prediction_timeout_seconds: float | None = None,
        prediction_backend: str | None = None,
    ):
        self.attempted = False
        self.prediction_timeout_seconds = prediction_timeout_seconds
        self.prediction_backend = prediction_backend

    def evaluate(
        self,
        model: Any,
        raw_features: pd.DataFrame,
        target: pd.Series,
        metric_fn: Callable,
    ) -> float:
        from gym.scoring import score_with_coercion

        if self.attempted:
            raise HiddenEvaluationAlreadyAttempted(
                "Hidden evaluation is limited to one attempt per agent outcome."
            )
        self.attempted = True
        try:
            model_payload = cloudpickle.dumps(model)
        except Exception as error:
            raise RuntimeError(f"Hidden candidate serialization failed: {error}") from error
        prediction = _predict_isolated(
            model_payload,
            raw_features,
            timeout_seconds=_prediction_timeout(self.prediction_timeout_seconds),
            prediction_backend=self.prediction_backend,
        )
        if not prediction["ok"]:
            raise RuntimeError(
                f"Hidden prediction failed ({prediction['error_type']}): "
                f"{prediction['error_message']}"
            )
        predictions = prediction["predictions"]
        if _prediction_length(predictions) != len(raw_features):
            raise ValueError("Hidden predictions have the wrong length.")
        if not _predictions_one_dimensional(predictions):
            raise ValueError("Hidden predictions must be one-dimensional.")
        if not _predictions_nan_free(predictions):
            raise ValueError("Hidden predictions contain null values.")
        score = score_with_coercion(metric_fn, target, predictions)
        if not np.isfinite(score):
            raise ValueError("Hidden evaluation metric is not finite.")
        return score


def _prediction_length(predictions: Any) -> int:
    try:
        return len(predictions)
    except TypeError:
        return -1


def _predictions_one_dimensional(predictions: Any) -> bool:
    try:
        return np.asarray(predictions).ndim == 1
    except Exception:
        return False


def _predictions_nan_free(predictions: Any) -> bool:
    try:
        values = np.asarray(predictions)
        if values.ndim == 0:
            return False
        return not bool(pd.isna(values).any())
    except Exception:
        return False


def _prediction_timeout(value: float | None) -> float:
    if value is None:
        raw = os.getenv(
            "AUTOVIBE_SUBMISSION_PREDICT_TIMEOUT_SECONDS",
            str(DEFAULT_PREDICTION_TIMEOUT_SECONDS),
        )
        try:
            value = float(raw)
        except ValueError:
            value = DEFAULT_PREDICTION_TIMEOUT_SECONDS
    if value <= 0:
        raise ValueError("prediction timeout must be positive")
    return value


def _predict_worker(connection: Any, model_payload: bytes, features_payload: bytes) -> None:
    try:
        model = cloudpickle.loads(model_payload)
        features = cloudpickle.loads(features_payload)
        predictions = model.predict(features)
        connection.send_bytes(
            cloudpickle.dumps({"ok": True, "predictions": predictions})
        )
    except BaseException as error:
        connection.send_bytes(
            cloudpickle.dumps(
                {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:2000],
                }
            )
        )
    finally:
        connection.close()


def _predict_isolated(
    model_payload: bytes,
    raw_features: pd.DataFrame,
    *,
    timeout_seconds: float,
    prediction_backend: str | None = None,
) -> dict[str, Any]:
    backend = (prediction_backend or "").strip().lower()
    if not backend:
        backend = os.getenv("AUTOVIBE_SUBMISSION_PREDICT_BACKEND", "").strip().lower()
    if not backend:
        backend = (
            "docker"
            if os.getenv("AUTOVIBE_KERNEL_BACKEND", "local").strip().lower()
            == "docker"
            else "process"
        )
    if backend == "docker":
        return _predict_docker(
            model_payload,
            raw_features,
            timeout_seconds=timeout_seconds,
        )
    if backend != "process":
        return {
            "ok": False,
            "error_type": "PredictionBackendError",
            "error_message": f"Unsupported prediction backend: {backend}",
        }
    return _predict_process(
        model_payload,
        raw_features,
        timeout_seconds=timeout_seconds,
    )


def _predict_process(
    model_payload: bytes,
    raw_features: pd.DataFrame,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        features_payload = cloudpickle.dumps(raw_features)
    except Exception as error:
        return {
            "ok": False,
            "error_type": type(error).__name__,
            "error_message": str(error)[:2000],
        }
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_predict_worker,
        args=(child, model_payload, features_payload),
        daemon=True,
    )
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=5)
            return {
                "ok": False,
                "error_type": "PredictionTimeout",
                "error_message": (
                    f"Candidate predict(X) exceeded {timeout_seconds:g} seconds."
                ),
            }
        try:
            result = cloudpickle.loads(parent.recv_bytes())
        except EOFError:
            process.join(timeout=5)
            return {
                "ok": False,
                "error_type": "PredictionProcessCrash",
                "error_message": (
                    "Candidate prediction process exited without a result "
                    f"(exit code {process.exitcode})."
                ),
            }
        process.join(timeout=5)
        return result
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def _predict_docker(
    model_payload: bytes,
    raw_features: pd.DataFrame,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        features_payload = cloudpickle.dumps(raw_features)
    except Exception as error:
        return {
            "ok": False,
            "error_type": type(error).__name__,
            "error_message": str(error)[:2000],
        }
    image = os.getenv("AUTOVIBE_SANDBOX_IMAGE", "autovibe-gym-sandbox:latest")
    container_name = f"autovibe-predict-{uuid.uuid4().hex[:16]}"
    with tempfile.TemporaryDirectory(prefix="autovibe-predict-") as directory:
        root = Path(directory)
        (root / "model.pkl").write_bytes(model_payload)
        (root / "features.pkl").write_bytes(features_payload)
        (root / "predict_worker.py").write_text(
            DOCKER_PREDICTION_WORKER,
            encoding="utf-8",
            newline="\n",
        )
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "128",
            "--memory",
            "2g",
            "--cpus",
            "2",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "--volume",
            f"{root.resolve()}:/work:rw",
            image,
            "python",
            "/work/predict_worker.py",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            return {
                "ok": False,
                "error_type": "PredictionTimeout",
                "error_message": (
                    f"Candidate predict(X) exceeded {timeout_seconds:g} seconds."
                ),
            }
        except OSError as error:
            return {
                "ok": False,
                "error_type": "PredictionBackendError",
                "error_message": str(error)[:2000],
            }
        result_path = root / "result.json"
        if completed.returncode != 0 or not result_path.is_file():
            detail = (completed.stderr or completed.stdout or "no process output")[-1000:]
            return {
                "ok": False,
                "error_type": "PredictionProcessCrash",
                "error_message": (
                    "Candidate Docker prediction exited without a result "
                    f"(exit code {completed.returncode}): {detail}"
                ),
            }
        return _decode_docker_result(result_path.read_text(encoding="utf-8"))


def _decode_docker_result(payload: str) -> dict[str, Any]:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"Non-finite JSON number is forbidden: {value}")

    try:
        result = json.loads(payload, parse_constant=reject_nonfinite)
    except Exception as error:
        return {
            "ok": False,
            "error_type": "PredictionContractError",
            "error_message": f"Cannot read isolated predictions: {error}",
        }
    if not isinstance(result, dict) or not isinstance(result.get("ok"), bool):
        return {
            "ok": False,
            "error_type": "PredictionContractError",
            "error_message": "Isolated evaluator returned an invalid result envelope.",
        }
    if result["ok"]:
        predictions = result.get("predictions")
        if not isinstance(predictions, list) or any(
            value is not None and not isinstance(value, (str, bool, int, float))
            for value in predictions
        ):
            return {
                "ok": False,
                "error_type": "PredictionContractError",
                "error_message": "Isolated predictions are not a JSON scalar vector.",
            }
        return {"ok": True, "predictions": predictions}
    if not isinstance(result.get("error_type"), str) or not isinstance(
        result.get("error_message"), str
    ):
        return {
            "ok": False,
            "error_type": "PredictionContractError",
            "error_message": "Isolated evaluator returned an invalid error envelope.",
        }
    return {
        "ok": False,
        "error_type": result["error_type"],
        "error_message": result["error_message"],
    }

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BYTES_PER_GB = 1024**3
BYTES_PER_MB = 1024**2


def bytes_to_gb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / BYTES_PER_GB, 4)


def gb_to_bytes(value: float | None) -> int | None:
    if value is None:
        return None
    return max(0, int(float(value) * BYTES_PER_GB))


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def run_command(argv: list[str], timeout: float = 5.0) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            argv,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", str(exc)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def detect_nvidia_smi() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    code, stdout, _stderr = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if code != 0 or not stdout:
        return []
    devices: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            devices.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "total_gb": round(float(parts[2]) / 1024.0, 4),
                    "used_gb": round(float(parts[3]) / 1024.0, 4),
                    "free_gb": round(float(parts[4]) / 1024.0, 4),
                    "source": "nvidia-smi",
                }
            )
        except ValueError:
            continue
    return devices


def detect_darwin_memory() -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {}
    code, stdout, _stderr = run_command(["sysctl", "-n", "hw.memsize"])
    if code != 0 or not stdout:
        return {}
    try:
        total = int(stdout.strip())
    except ValueError:
        return {}
    return {
        "unified_memory_total_gb": bytes_to_gb(total),
        "source": "sysctl hw.memsize",
    }


@dataclass
class ProbeSummary:
    status: str
    mode: str
    backend: str | None
    started_at: str
    ended_at: str | None = None
    elapsed_seconds: float | None = None
    peak_gb: float | None = None
    device: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def finish(self, started_monotonic: float) -> "ProbeSummary":
        self.ended_at = now_iso()
        self.elapsed_seconds = round(time.monotonic() - started_monotonic, 3)
        peaks = [
            sample.get("peak_gb")
            for sample in self.samples
            if isinstance(sample.get("peak_gb"), (int, float))
        ]
        if peaks:
            self.peak_gb = round(max(float(value) for value in peaks), 4)
        return self


class StressBackend:
    name = "base"

    def availability(self) -> tuple[bool, str]:
        raise NotImplementedError

    def device_info(self) -> dict[str, Any]:
        return {}

    def total_bytes(self) -> int | None:
        return None

    def free_bytes(self) -> int | None:
        return None

    def reset_peak(self) -> None:
        return None

    def allocate(self, num_bytes: int) -> None:
        raise NotImplementedError

    def synchronize(self) -> None:
        return None

    def sample(self) -> dict[str, Any]:
        return {}

    def cleanup(self) -> None:
        return None


class TorchCudaBackend(StressBackend):
    name = "torch-cuda"

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self.torch = None
        self.device = None
        self.tensors: list[Any] = []

    def _load(self) -> Any:
        if self.torch is None:
            import torch

            self.torch = torch
            self.device = torch.device(f"cuda:{self.device_index}")
        return self.torch

    def availability(self) -> tuple[bool, str]:
        try:
            torch = self._load()
            if not torch.cuda.is_available():
                return False, "torch is installed but CUDA is not available"
            count = torch.cuda.device_count()
            if self.device_index >= count:
                return False, f"CUDA device {self.device_index} is out of range; found {count}"
            return True, "available"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def device_info(self) -> dict[str, Any]:
        torch = self._load()
        props = torch.cuda.get_device_properties(self.device_index)
        free, total = torch.cuda.mem_get_info(self.device_index)
        return {
            "name": props.name,
            "index": self.device_index,
            "total_gb": bytes_to_gb(total),
            "free_gb": bytes_to_gb(free),
            "capability": ".".join(str(part) for part in props.major_minor)
            if hasattr(props, "major_minor")
            else f"{props.major}.{props.minor}",
            "source": "torch.cuda",
        }

    def total_bytes(self) -> int | None:
        torch = self._load()
        _free, total = torch.cuda.mem_get_info(self.device_index)
        return int(total)

    def free_bytes(self) -> int | None:
        torch = self._load()
        free, _total = torch.cuda.mem_get_info(self.device_index)
        return int(free)

    def reset_peak(self) -> None:
        torch = self._load()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device_index)

    def allocate(self, num_bytes: int) -> None:
        torch = self._load()
        element_size = 2
        elements = max(1, num_bytes // element_size)
        tensor = torch.empty((elements,), dtype=torch.float16, device=self.device)
        tensor.fill_(1)
        self.tensors.append(tensor)
        self.synchronize()

    def synchronize(self) -> None:
        self._load().cuda.synchronize(self.device_index)

    def sample(self) -> dict[str, Any]:
        torch = self._load()
        free, total = torch.cuda.mem_get_info(self.device_index)
        return {
            "active_gb": bytes_to_gb(torch.cuda.memory_allocated(self.device_index)),
            "reserved_gb": bytes_to_gb(torch.cuda.memory_reserved(self.device_index)),
            "peak_gb": bytes_to_gb(torch.cuda.max_memory_allocated(self.device_index)),
            "free_gb": bytes_to_gb(free),
            "total_gb": bytes_to_gb(total),
            "measurement_source": "torch.cuda.max_memory_allocated",
        }

    def cleanup(self) -> None:
        self.tensors.clear()
        gc.collect()
        torch = self._load()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device_index)


class TorchMPSBackend(StressBackend):
    name = "torch-mps"

    def __init__(self) -> None:
        self.torch = None
        self.device = None
        self.tensors: list[Any] = []
        self.max_driver_bytes = 0
        self.max_active_bytes = 0

    def _load(self) -> Any:
        if self.torch is None:
            import torch

            self.torch = torch
            self.device = torch.device("mps")
        return self.torch

    def availability(self) -> tuple[bool, str]:
        try:
            torch = self._load()
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                return False, "torch is installed but MPS is not available"
            return True, "available"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def device_info(self) -> dict[str, Any]:
        torch = self._load()
        recommended = None
        if hasattr(torch, "mps") and hasattr(torch.mps, "recommended_max_memory"):
            try:
                recommended = int(torch.mps.recommended_max_memory())
            except Exception:
                recommended = None
        info = {
            "name": "Apple Metal Performance Shaders",
            "recommended_max_memory_gb": bytes_to_gb(recommended),
            "source": "torch.mps",
        }
        info.update(detect_darwin_memory())
        return info

    def total_bytes(self) -> int | None:
        torch = self._load()
        if hasattr(torch, "mps") and hasattr(torch.mps, "recommended_max_memory"):
            try:
                return int(torch.mps.recommended_max_memory())
            except Exception:
                return None
        return None

    def reset_peak(self) -> None:
        self.max_driver_bytes = 0
        self.max_active_bytes = 0
        torch = self._load()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def allocate(self, num_bytes: int) -> None:
        torch = self._load()
        element_size = 2
        elements = max(1, num_bytes // element_size)
        tensor = torch.empty((elements,), dtype=torch.float16, device=self.device)
        tensor.fill_(1)
        self.tensors.append(tensor)
        self.synchronize()

    def synchronize(self) -> None:
        torch = self._load()
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    def sample(self) -> dict[str, Any]:
        torch = self._load()
        active = None
        driver = None
        if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            active = int(torch.mps.current_allocated_memory())
            self.max_active_bytes = max(self.max_active_bytes, active)
        if hasattr(torch, "mps") and hasattr(torch.mps, "driver_allocated_memory"):
            driver = int(torch.mps.driver_allocated_memory())
            self.max_driver_bytes = max(self.max_driver_bytes, driver)
        peak = max(self.max_driver_bytes, self.max_active_bytes)
        return {
            "active_gb": bytes_to_gb(active),
            "driver_allocated_gb": bytes_to_gb(driver),
            "peak_gb": bytes_to_gb(peak),
            "recommended_max_memory_gb": bytes_to_gb(self.total_bytes()),
            "measurement_source": "max observed torch.mps driver/current allocation",
        }

    def cleanup(self) -> None:
        self.tensors.clear()
        gc.collect()
        torch = self._load()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


class MLXBackend(StressBackend):
    name = "mlx"

    def __init__(self) -> None:
        self.mx = None
        self.tensors: list[Any] = []
        self.max_observed_bytes = 0

    def _load(self) -> Any:
        if self.mx is None:
            import mlx.core as mx

            self.mx = mx
        return self.mx

    def _metal(self) -> Any:
        mx = self._load()
        return getattr(mx, "metal", None)

    def availability(self) -> tuple[bool, str]:
        try:
            metal = self._metal()
            if metal is None:
                return False, "mlx is installed but mlx.core.metal is not available"
            return True, "available"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def _device_info_raw(self) -> dict[str, Any]:
        metal = self._metal()
        if metal is None or not hasattr(metal, "device_info"):
            return {}
        try:
            info = metal.device_info()
        except Exception:
            return {}
        return info if isinstance(info, dict) else {}

    @staticmethod
    def _pick_memory_bytes(info: dict[str, Any]) -> int | None:
        keys = [
            "max_recommended_working_set_size",
            "recommended_max_working_set_size",
            "memory_size",
            "total_memory",
            "unified_memory",
        ]
        for key in keys:
            value = info.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
        return None

    def device_info(self) -> dict[str, Any]:
        raw = self._device_info_raw()
        total = self._pick_memory_bytes(raw)
        compact = {
            "name": raw.get("device_name") or raw.get("name") or "Apple MLX Metal",
            "recommended_or_total_gb": bytes_to_gb(total),
            "source": "mlx.core.metal",
        }
        if raw:
            compact["raw_keys"] = sorted(str(key) for key in raw.keys())
        compact.update(detect_darwin_memory())
        return compact

    def total_bytes(self) -> int | None:
        return self._pick_memory_bytes(self._device_info_raw())

    def reset_peak(self) -> None:
        metal = self._metal()
        self.max_observed_bytes = 0
        if metal is not None and hasattr(metal, "clear_cache"):
            try:
                metal.clear_cache()
            except Exception:
                pass
        if metal is not None and hasattr(metal, "reset_peak_memory"):
            try:
                metal.reset_peak_memory()
            except Exception:
                pass

    def allocate(self, num_bytes: int) -> None:
        mx = self._load()
        element_size = 2
        elements = max(1, num_bytes // element_size)
        tensor = mx.zeros((elements,), dtype=mx.float16)
        mx.eval(tensor)
        self.tensors.append(tensor)

    def sample(self) -> dict[str, Any]:
        metal = self._metal()
        active = None
        cache = None
        peak = None
        if metal is not None and hasattr(metal, "get_active_memory"):
            try:
                active = int(metal.get_active_memory())
            except Exception:
                active = None
        if metal is not None and hasattr(metal, "get_cache_memory"):
            try:
                cache = int(metal.get_cache_memory())
            except Exception:
                cache = None
        if metal is not None and hasattr(metal, "get_peak_memory"):
            try:
                peak = int(metal.get_peak_memory())
            except Exception:
                peak = None
        estimated_held = sum(getattr(tensor, "nbytes", 0) for tensor in self.tensors)
        observed = max(value or 0 for value in [active, cache, peak, estimated_held])
        self.max_observed_bytes = max(self.max_observed_bytes, int(observed))
        return {
            "active_gb": bytes_to_gb(active),
            "cache_gb": bytes_to_gb(cache),
            "peak_gb": bytes_to_gb(peak if peak is not None else self.max_observed_bytes),
            "estimated_held_gb": bytes_to_gb(estimated_held),
            "measurement_source": "mlx.core.metal peak memory when available, otherwise held tensor bytes",
        }

    def cleanup(self) -> None:
        self.tensors.clear()
        gc.collect()
        metal = self._metal()
        if metal is not None and hasattr(metal, "clear_cache"):
            try:
                metal.clear_cache()
            except Exception:
                pass


def environment_info() -> dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "repo_root": str(REPO_ROOT),
        "nvidia_smi": detect_nvidia_smi(),
    }
    darwin = detect_darwin_memory()
    if darwin:
        info["darwin_memory"] = darwin
    return info


def backend_candidates(args: argparse.Namespace) -> list[StressBackend]:
    requested = args.backend
    if requested == "mlx":
        return [MLXBackend()]
    if requested == "torch-cuda":
        return [TorchCudaBackend(args.device_index)]
    if requested == "torch-mps":
        return [TorchMPSBackend()]
    return [MLXBackend(), TorchCudaBackend(args.device_index), TorchMPSBackend()]


def pick_backend(args: argparse.Namespace) -> tuple[StressBackend | None, list[str]]:
    reasons: list[str] = []
    for backend in backend_candidates(args):
        ok, reason = backend.availability()
        reasons.append(f"{backend.name}: {reason}")
        if ok:
            return backend, reasons
    return None, reasons


def resolve_allocation_limit_bytes(backend: StressBackend, args: argparse.Namespace) -> tuple[int, list[str]]:
    notes: list[str] = []
    hard_cap = gb_to_bytes(args.max_gb)
    total = backend.total_bytes()
    free = backend.free_bytes()
    min_free = gb_to_bytes(args.min_free_gb) or 0

    if args.until_failure:
        if free is not None:
            limit = max(0, free - min_free)
            notes.append("until-failure enabled; using current free memory minus min-free as safety limit")
        elif hard_cap is not None:
            limit = hard_cap
            notes.append("until-failure enabled; no free-memory API, using --max-gb as safety limit")
        else:
            limit = gb_to_bytes(args.unknown_limit_gb) or 0
            notes.append("until-failure enabled but memory size is unknown; using --unknown-limit-gb")
        return limit, notes

    if total is not None:
        target_device_used = int(total * args.target_utilization)
        if free is not None:
            already_used = max(0, total - free)
            cap_by_target = max(0, target_device_used - already_used)
            cap_by_free = max(0, free - min_free)
            limit = min(cap_by_target, cap_by_free)
            notes.append("limit derived from total/free memory, target utilization, and min-free")
        else:
            limit = target_device_used
            notes.append("limit derived from reported total/recommended memory and target utilization")
    else:
        limit = gb_to_bytes(args.unknown_limit_gb) or 0
        notes.append("memory size is unknown; using --unknown-limit-gb")

    if hard_cap is not None:
        limit = min(limit, hard_cap)
        notes.append("--max-gb applied as a hard process allocation cap")
    return max(0, limit), notes


def run_allocation_probe(args: argparse.Namespace) -> ProbeSummary:
    started = time.monotonic()
    summary = ProbeSummary(
        status="running",
        mode="allocation",
        backend=None,
        started_at=now_iso(),
        environment=environment_info(),
    )
    backend, reasons = pick_backend(args)
    summary.notes.extend(reasons)
    if backend is None:
        summary.status = "skipped_no_stress_backend"
        summary.error = "No supported allocation backend is available"
        return summary.finish(started)

    summary.backend = backend.name
    summary.device = backend.device_info()
    limit, limit_notes = resolve_allocation_limit_bytes(backend, args)
    summary.notes.extend(limit_notes)
    summary.notes.append(f"allocation_limit_gb={bytes_to_gb(limit)}")

    if limit <= 0:
        summary.status = "skipped_zero_limit"
        summary.error = "Resolved allocation limit is zero"
        return summary.finish(started)

    chunk_bytes = max(BYTES_PER_MB, int(args.chunk_mb * BYTES_PER_MB))
    held_bytes = 0
    backend.reset_peak()
    try:
        while held_bytes < limit:
            requested = min(chunk_bytes, limit - held_bytes)
            if requested <= 0:
                break
            try:
                backend.allocate(requested)
            except Exception as exc:
                summary.status = "allocation_failed"
                summary.error = f"{type(exc).__name__}: {exc}"
                break
            held_bytes += requested
            sample = backend.sample()
            sample.update(
                {
                    "phase": "allocation",
                    "step": len(summary.samples) + 1,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "requested_held_gb": bytes_to_gb(held_bytes),
                    "requested_chunk_gb": bytes_to_gb(requested),
                }
            )
            summary.samples.append(sample)
            if args.sleep_per_step > 0:
                time.sleep(args.sleep_per_step)
        if args.hold_seconds > 0:
            time.sleep(args.hold_seconds)
            sample = backend.sample()
            sample.update(
                {
                    "phase": "hold",
                    "step": len(summary.samples) + 1,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "requested_held_gb": bytes_to_gb(held_bytes),
                    "requested_chunk_gb": 0.0,
                }
            )
            summary.samples.append(sample)
        if summary.status == "running":
            summary.status = "completed"
    finally:
        if not args.keep_allocated:
            backend.cleanup()
    return summary.finish(started)


def token_targets(start_tokens: int, max_tokens: int, growth_factor: float) -> list[int]:
    start_tokens = max(1, int(start_tokens))
    max_tokens = max(start_tokens, int(max_tokens))
    growth_factor = max(1.05, float(growth_factor))
    values: list[int] = []
    current = start_tokens
    while current < max_tokens:
        values.append(current)
        next_value = int(math.ceil(current * growth_factor / 256.0) * 256)
        current = max(current + 1, next_value)
    if not values or values[-1] != max_tokens:
        values.append(max_tokens)
    return values


def pressure_text(word_count: int) -> str:
    parts = [
        "This is a local peak memory pressure probe.",
        "The following synthetic tokens are intentionally repetitive and factual-neutral.",
    ]
    filler = [f"probe_token_{index:06d}" for index in range(max(1, word_count))]
    parts.append(" ".join(filler))
    parts.append("Return exactly one short sentence.")
    return "\n".join(parts)


def build_llm_prompt_for_target(session: Any, target_tokens: int) -> tuple[list[dict], str, int]:
    from chat_backend import build_user_message

    word_count = max(16, target_tokens)
    messages: list[dict] = [session._system_message, build_user_message(pressure_text(word_count))]
    prompt_text = session.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_tokens = session._estimate_prompt_tokens(messages, prompt_text)

    for _ in range(6):
        if prompt_tokens >= int(target_tokens * 0.98):
            break
        deficit = target_tokens - prompt_tokens
        word_count += max(32, int(deficit * 1.15))
        messages = [session._system_message, build_user_message(pressure_text(word_count))]
        prompt_text = session.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = session._estimate_prompt_tokens(messages, prompt_text)
    return messages, prompt_text, prompt_tokens


def run_llm_prefill_probe(args: argparse.Namespace) -> ProbeSummary:
    started = time.monotonic()
    summary = ProbeSummary(
        status="running",
        mode="llm-prefill",
        backend="project-chat-backend",
        started_at=now_iso(),
        environment=environment_info(),
    )
    try:
        from chat_backend import ChatSession, SchedulerConfig
    except Exception as exc:
        summary.status = "skipped_no_llm_backend"
        summary.error = f"Cannot import project chat backend: {type(exc).__name__}: {exc}"
        return summary.finish(started)

    model_path = Path(args.model_path).expanduser().resolve()
    config = SchedulerConfig(
        cage_budget_gb=None,
        prefill_context_cap_tokens=None,
        structured_compression_enabled=False,
    )
    session = ChatSession(model_path=model_path, scheduler_config=config)
    session.trace_prefill = False
    try:
        load_started = time.monotonic()
        session.load()
        summary.device = {
            "model_path": str(model_path),
            "load_seconds": round(time.monotonic() - load_started, 3),
            "source": "ChatSession.load",
        }
    except Exception as exc:
        summary.status = "skipped_no_llm_backend"
        summary.error = f"Cannot load local model runtime: {type(exc).__name__}: {exc}"
        if args.debug:
            summary.notes.append(traceback.format_exc())
        return summary.finish(started)

    targets = token_targets(args.start_tokens, args.max_tokens, args.growth_factor)
    summary.notes.append(f"token_targets={targets}")
    best_peak = 0.0

    for index, target in enumerate(targets, start=1):
        try:
            _messages, prompt_text, prompt_tokens = build_llm_prompt_for_target(session, target)
            prompt_cache_state = (
                session._prompt_cache_state_cls()
                if session._prompt_cache_state_cls is not None
                else None
            )
            vision_cache = (
                session._vision_feature_cache_cls(max_size=1)
                if session._vision_feature_cache_cls is not None
                else None
            )
            run_started = time.monotonic()
            max_peak = 0.0
            max_prompt_tps = 0.0
            final_prompt_tokens = prompt_tokens
            final_generation_tokens = 0
            for chunk in session._vlm_stream_generate(
                session.model,
                session.processor,
                prompt=prompt_text,
                image=None,
                max_tokens=args.decode_tokens,
                prefill_step_size=args.prefill_step_size,
                prompt_cache_state=prompt_cache_state,
                vision_cache=vision_cache,
            ):
                max_peak = max(max_peak, float(getattr(chunk, "peak_memory", 0.0) or 0.0))
                max_prompt_tps = max(max_prompt_tps, float(getattr(chunk, "prompt_tps", 0.0) or 0.0))
                final_prompt_tokens = int(getattr(chunk, "prompt_tokens", final_prompt_tokens) or final_prompt_tokens)
                final_generation_tokens = int(getattr(chunk, "generation_tokens", final_generation_tokens) or 0)
            elapsed = time.monotonic() - run_started
            best_peak = max(best_peak, max_peak)
            sample = {
                "phase": "llm-prefill",
                "step": index,
                "target_tokens": target,
                "prompt_tokens": final_prompt_tokens,
                "generation_tokens": final_generation_tokens,
                "prefill_step_size": args.prefill_step_size,
                "prompt_tps": round(max_prompt_tps, 4),
                "run_seconds": round(elapsed, 3),
                "peak_gb": round(max_peak, 4),
                "best_peak_gb": round(best_peak, 4),
                "measurement_source": "stream_generate chunk.peak_memory",
            }
            summary.samples.append(sample)
            if args.stop_peak_gb is not None and max_peak >= args.stop_peak_gb:
                summary.notes.append(f"stopped because peak_gb reached --stop-peak-gb={args.stop_peak_gb}")
                break
            if args.sleep_per_step > 0:
                time.sleep(args.sleep_per_step)
        except Exception as exc:
            summary.status = "llm_prefill_failed"
            summary.error = f"{type(exc).__name__}: {exc}"
            if args.debug:
                summary.notes.append(traceback.format_exc())
            break

    if summary.status == "running":
        summary.status = "completed" if summary.samples else "skipped_no_samples"
    return summary.finish(started)


def run_auto_probe(args: argparse.Namespace) -> ProbeSummary:
    llm_summary = run_llm_prefill_probe(args)
    if llm_summary.status in {"completed", "llm_prefill_failed"} and llm_summary.samples:
        return llm_summary
    allocation_summary = run_allocation_probe(args)
    allocation_summary.notes.insert(
        0,
        f"llm-prefill fallback reason: {llm_summary.status}; {llm_summary.error or 'no error'}",
    )
    return allocation_summary


def print_human_summary(summary: ProbeSummary) -> None:
    print(f"status: {summary.status}")
    print(f"mode: {summary.mode}")
    print(f"backend: {summary.backend or 'none'}")
    if summary.peak_gb is not None:
        print(f"peak_gb: {summary.peak_gb}")
    if summary.device:
        print(f"device: {json.dumps(summary.device, ensure_ascii=False)}")
    if summary.error:
        print(f"error: {summary.error}")
    if summary.notes:
        print("notes:")
        for note in summary.notes:
            print(f"  - {note}")
    if summary.samples:
        last = summary.samples[-1]
        print(f"samples: {len(summary.samples)}")
        print(f"last_sample: {json.dumps(last, ensure_ascii=False)}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe peak GPU/VRAM pressure for this machine. "
            "Auto mode first tries the local LLM prefill path, then falls back to generic allocation."
        )
    )
    parser.add_argument("--mode", choices=["auto", "llm-prefill", "allocation"], default="auto")
    parser.add_argument("--backend", choices=["auto", "mlx", "torch-cuda", "torch-mps"], default="auto")
    parser.add_argument("--device-index", type=int, default=0, help="CUDA device index for torch-cuda.")
    parser.add_argument("--model-path", default=str(REPO_ROOT / "model"), help="Model path for llm-prefill mode.")
    parser.add_argument("--start-tokens", type=int, default=1024, help="First target prompt size for llm-prefill.")
    parser.add_argument("--max-tokens", type=int, default=32768, help="Largest target prompt size for llm-prefill.")
    parser.add_argument("--growth-factor", type=float, default=1.6, help="Token target growth factor.")
    parser.add_argument("--decode-tokens", type=int, default=1, help="Decode tokens after prefill in llm-prefill mode.")
    parser.add_argument("--prefill-step-size", type=int, default=256)
    parser.add_argument("--chunk-mb", type=int, default=256, help="Allocation chunk size for allocation mode.")
    parser.add_argument("--target-utilization", type=float, default=0.80, help="Target device utilization for allocation mode.")
    parser.add_argument("--max-gb", type=float, default=None, help="Hard cap for process allocation in allocation mode.")
    parser.add_argument("--min-free-gb", type=float, default=3.0, help="Free-memory reserve when the backend reports free memory.")
    parser.add_argument("--unknown-limit-gb", type=float, default=4.0, help="Safety cap when total memory is unknown.")
    parser.add_argument("--until-failure", action="store_true", help="Allocate until failure within the safety limit.")
    parser.add_argument("--hold-seconds", type=float, default=2.0, help="Hold pressure before final sample.")
    parser.add_argument("--sleep-per-step", type=float, default=0.0)
    parser.add_argument("--stop-peak-gb", type=float, default=None, help="Stop llm-prefill mode once this peak is observed.")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to write JSON result.")
    parser.add_argument("--json", action="store_true", help="Print full JSON to stdout.")
    parser.add_argument("--keep-allocated", action="store_true", help="Debug only: do not release allocation tensors before exit.")
    parser.add_argument("--require-stress", action="store_true", help="Return non-zero when no real stress backend runs.")
    parser.add_argument("--debug", action="store_true", help="Include tracebacks in JSON notes.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not 0 < args.target_utilization <= 0.98:
        print("--target-utilization must be in (0, 0.98]", file=sys.stderr)
        return 2
    if args.mode == "auto":
        summary = run_auto_probe(args)
    elif args.mode == "llm-prefill":
        summary = run_llm_prefill_probe(args)
    else:
        summary = run_allocation_probe(args)

    payload = {
        "status": summary.status,
        "mode": summary.mode,
        "backend": summary.backend,
        "started_at": summary.started_at,
        "ended_at": summary.ended_at,
        "elapsed_seconds": summary.elapsed_seconds,
        "peak_gb": summary.peak_gb,
        "device": summary.device,
        "environment": summary.environment,
        "samples": summary.samples,
        "notes": summary.notes,
        "error": summary.error,
    }

    if args.json_out:
        output_path = Path(args.json_out).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_human_summary(summary)

    if summary.status in {"completed", "allocation_failed", "llm_prefill_failed"}:
        return 0
    if not args.require_stress:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

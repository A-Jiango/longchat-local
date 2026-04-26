from __future__ import annotations

import copy
import inspect
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generator

from compression_cache import (
    BacktrackingRetrievalResult,
    CompressionPool,
    CompressionTurnReport,
    LLMResponse,
    LLMRunStats,
    TurnSemanticDAG,
)


MODEL_PATH = Path(__file__).resolve().with_name("model")
DEFAULT_MODEL_DISPLAY_NAME = "LLM"
SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.md")
LEGACY_SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.txt")
DEFAULT_SYSTEM_PROMPT = """You are a locally running AI assistant.

Follow these response rules:
1. Use the language of the user's initial input by default; switch only when the user explicitly requests another language.
2. Prefer direct, clear, actionable answers and avoid empty wording.
3. Be honest: state uncertainty clearly and do not invent facts, sources, data, or conclusions.
4. Unless the user explicitly asks, do not show chain-of-thought or long reasoning; give the conclusion, necessary explanation, and steps.
5. For math, use standard LaTeX:
   - Inline formulas use $...$
   - Block formulas use $$...$$
   - Prefer standard LaTeX commands and avoid Unicode pseudo-math characters
6. For images, tables, screenshots, or OCR:
   - If content is unclear, state the uncertainty
   - Do not treat unreadable content as confirmed fact
7. Keep the style concise, professional, and stable, with little exaggeration or empty politeness.
8. If bullet points fit the question, use concise Markdown lists, but do not add structure just for formatting.
"""
MAX_TOKENS = 2048
MAX_ROUNDS = 4
PREFILL_STEP_SIZE = 256
COMPRESSION_PREFILL_STEP_SIZE = 256
VISION_CACHE_SIZE = 8
DEFAULT_PROMPT_TPS = 900.0
MIN_PREFILL_ESTIMATE_SECONDS = 0.45
MAX_PREFILL_ESTIMATE_SECONDS = 12.0
DEFAULT_KV_COST_PER_TOKEN_GB = 0.00002
DEFAULT_IMAGE_TURN_PENALTY_GB = 0.15
DEFAULT_EPSILON_SURVIVAL_GB = 0.15
DEFAULT_AUTO_PREFILL_SECONDS = 12.0
MIN_AUTO_PREFILL_CONTEXT_TOKENS = 1024
MIN_RECENT_MEMORY_ITEMS = 2
COMPRESSION_MAX_PASSES = 256
DEFAULT_STRUCTURED_COMPRESSION_ENABLED = True
DEFAULT_STRUCTURED_RECENT_TURNS = 0
DEFAULT_STRUCTURED_MAX_DIRTY_BLOCKS = 1
DEFAULT_STRUCTURED_TARGET_TOKENS_PER_BLOCK = 180
DEFAULT_COMPRESSION_UPDATE_WAIT_SECONDS = 20.0
DEFAULT_COMPRESSION_UPDATE_PREFILL_TPS = 120.0
DEFAULT_COMPRESSION_UPDATE_RAMP_TURNS = 20


def _load_system_prompt() -> str:
    for path in (SYSTEM_PROMPT_PATH, LEGACY_SYSTEM_PROMPT_PATH):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        except OSError:
            continue
        if text:
            return text
    return DEFAULT_SYSTEM_PROMPT.strip()


SYSTEM_PROMPT = _load_system_prompt()


class RuntimePhase(Enum):
    IDLE = "idle"
    PREFILL = "prefill"
    DECODE = "decode"
    COMPRESSING = "compressing"
    CLEARING = "clearing"
    ERROR = "error"


class ContextBudgetError(RuntimeError):
    pass


@dataclass
class TurnStats:
    prompt_tokens: int = 0
    generation_tokens: int = 0
    total_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory: float = 0.0


@dataclass
class TurnChunk:
    text: str
    stats: TurnStats


@dataclass
class TurnResult:
    text: str
    stats: TurnStats


@dataclass(frozen=True)
class UserTurn:
    text: str
    image_path: str | None = None


@dataclass
class PrefillEstimate:
    seconds: float
    prompt_tokens: int


@dataclass
class PrefillProgress:
    prompt_tokens: int
    total_prompt_tokens: int
    prompt_tps: float


@dataclass
class CompressionProgress:
    message: str = "正在压缩上下文..."
    tone: str = "compression"


@dataclass
class SchedulerConfig:
    cage_budget_gb: float | None = None
    prefill_context_cap_tokens: int | None = None
    locale: str = "zh-CN"
    structured_compression_enabled: bool = False
    structured_recent_turns: int = DEFAULT_STRUCTURED_RECENT_TURNS
    structured_max_dirty_blocks: int = DEFAULT_STRUCTURED_MAX_DIRTY_BLOCKS
    structured_target_tokens_per_block: int = DEFAULT_STRUCTURED_TARGET_TOKENS_PER_BLOCK
    compression_update_wait_seconds: float = DEFAULT_COMPRESSION_UPDATE_WAIT_SECONDS


@dataclass
class ModelProfile:
    max_position_embeddings: int | None = None
    num_hidden_layers: int | None = None
    quant_bits: int | None = None
    has_vision: bool = False


@dataclass
class HardwareProfile:
    base_resident_gb: float = 0.0
    safety_margin_gb: float = DEFAULT_EPSILON_SURVIVAL_GB
    dynamic_budget_gb: float = math.inf
    dynamic_ceiling_estimate_gb: float = 0.0


@dataclass
class RuntimeTelemetry:
    prompt_tps_ema: float = DEFAULT_PROMPT_TPS
    compression_prompt_tps_ema: float = DEFAULT_COMPRESSION_UPDATE_PREFILL_TPS
    kv_cost_per_token_gb: float = DEFAULT_KV_COST_PER_TOKEN_GB
    image_turn_penalty_gb: float = DEFAULT_IMAGE_TURN_PENALTY_GB
    last_peak_memory_gb: float = 0.0


@dataclass
class PressureSnapshot:
    projected_usage_gb: float
    projected_prompt_tokens: int
    pressure_score: float
    urgency_score: float
    compression_mode: str
    debounce_ms: int
    effective_prefill_context_cap_tokens: int


@dataclass
class PreparedTurn:
    turn: UserTurn
    revision: int
    prompt_messages: list[dict]
    prompt_text: str
    prompt_tokens: int
    pressure: PressureSnapshot
    retrieval_context: str = ""
    retrieval_result: BacktrackingRetrievalResult | None = None


@dataclass
class CompressionOutcome:
    prompt_cache_invalidated: bool = False
    vision_cache_invalidated: bool = False
    compressed: bool = False
    structured_report: CompressionTurnReport | None = None


@dataclass
class MemoryItem:
    role: str
    content: list[dict[str, Any]]
    timestamp: float
    token_size: int
    weight: float = 1.0
    has_image: bool = False

    def to_message(self) -> dict:
        return {"role": self.role, "content": copy.deepcopy(self.content)}


@dataclass
class ContextMemoryState:
    system_message: dict
    short_term: list[MemoryItem] = field(default_factory=list)
    mid_term: list[MemoryItem] = field(default_factory=list)
    long_term: list[MemoryItem] = field(default_factory=list)
    initial_user_language: str | None = None
    explicit_response_language: str | None = None
    revision: int = 0


PromptBuilder = Callable[[list[dict]], str]
TokenEstimator = Callable[[list[dict], str], int]


def _load_local_env_values() -> dict[str, str]:
    env_path = Path(__file__).with_name(".env")
    values: dict[str, str] = {}
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _get_config_value(name: str) -> str | None:
    value = os.getenv(name)
    if value is not None:
        return value
    return _load_local_env_values().get(name)


def _clean_model_display_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    name = re.sub(r"\s+", " ", value).strip()
    if not name:
        return None
    for separator in (" — ", " – ", " | "):
        if separator in name:
            name = name.split(separator, 1)[0].strip()
    return name or None


def _format_jang_source_model_name(name: str, metadata: dict[str, Any]) -> str:
    display_name = name
    capabilities = metadata.get("capabilities")
    if isinstance(capabilities, dict):
        modality = str(capabilities.get("modality", "")).lower()
        if modality == "vision" and "vl" not in display_name.lower():
            display_name = re.sub(r"(?=-\d+(?:\.\d+)?b\b)", "-VL", display_name, count=1, flags=re.IGNORECASE)
    return display_name


def _read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def detect_model_display_name(model_path: Path | str) -> str:
    path = Path(model_path)

    try:
        jang_config = _read_json_file(path / "jang_config.json")
    except (OSError, json.JSONDecodeError):
        jang_config = None
    if isinstance(jang_config, dict):
        source_model = jang_config.get("source_model")
        if isinstance(source_model, dict):
            source_name = _clean_model_display_name(source_model.get("name"))
            if source_name:
                return _format_jang_source_model_name(source_name, jang_config)

    try:
        config = _read_json_file(path / "config.json")
    except (OSError, json.JSONDecodeError):
        config = None
    if isinstance(config, dict):
        for key in ("model_name", "name", "_name_or_path", "name_or_path"):
            config_name = _clean_model_display_name(config.get(key))
            if config_name:
                return config_name.rsplit("/", 1)[-1]

    readme_path = path / "README.md"
    try:
        for line in readme_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                readme_title = _clean_model_display_name(line[2:])
                if readme_title:
                    return readme_title
    except OSError:
        pass

    directory_name = _clean_model_display_name(path.name)
    if directory_name and directory_name.lower() not in {"model", "models", "llm"}:
        return directory_name
    return DEFAULT_MODEL_DISPLAY_NAME


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized or normalized == "null":
        return None
    return float(normalized)


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized or normalized == "null":
        return None
    return int(normalized)


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    normalized = value.strip().lower()
    if not normalized or normalized == "null":
        return default
    return int(normalized)


def scheduler_config_from_env() -> SchedulerConfig:
    return SchedulerConfig(
        cage_budget_gb=_parse_optional_float(_get_config_value("LLM_CONTEXT_K_GB")),
        prefill_context_cap_tokens=_parse_optional_int(_get_config_value("LLM_PREFILL_CONTEXT_CAP_TOKENS")),
        locale=(_get_config_value("LLM_LOCALE") or "zh-CN").strip() or "zh-CN",
        structured_compression_enabled=_parse_bool(
            _get_config_value("LLM_STRUCTURED_COMPRESSION"),
            DEFAULT_STRUCTURED_COMPRESSION_ENABLED,
        ),
        structured_recent_turns=max(
            0,
            _parse_int(_get_config_value("LLM_STRUCTURED_RECENT_TURNS"), DEFAULT_STRUCTURED_RECENT_TURNS),
        ),
        structured_max_dirty_blocks=max(
            1,
            _parse_int(_get_config_value("LLM_STRUCTURED_MAX_DIRTY_BLOCKS"), DEFAULT_STRUCTURED_MAX_DIRTY_BLOCKS),
        ),
        structured_target_tokens_per_block=max(
            32,
            _parse_int(
                _get_config_value("LLM_STRUCTURED_TARGET_TOKENS_PER_BLOCK"),
                DEFAULT_STRUCTURED_TARGET_TOKENS_PER_BLOCK,
            ),
        ),
        compression_update_wait_seconds=max(
            1.0,
            _parse_optional_float(_get_config_value("LLM_COMPRESSION_UPDATE_WAIT_SECONDS"))
            or DEFAULT_COMPRESSION_UPDATE_WAIT_SECONDS,
        ),
    )


def build_text_message(role: str, text: str) -> dict:
    return {
        "role": role,
        "content": [{"type": "text", "text": text}],
    }


def build_user_message(text: str, image_path: str | None = None) -> dict:
    content = []
    if image_path:
        content.append({"type": "image", "image": image_path})
    if text:
        content.append({"type": "text", "text": text})
    return {
        "role": "user",
        "content": content,
    }


def detect_initial_user_language(text: str) -> str | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    cjk_chars = sum(1 for char in cleaned if "\u4e00" <= char <= "\u9fff")
    latin_words = len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", cleaned))
    if cjk_chars == 0 and latin_words > 0:
        return "English"
    if cjk_chars > 0 and cjk_chars * 2 >= max(1, latin_words):
        return "Chinese"
    if latin_words > 0:
        return "English"
    return None


def _is_negated_language_match(text: str, start: int) -> bool:
    prefix = text[max(0, start - 14) : start].lower()
    return bool(re.search(r"(不要|别|不必|无需|不用|不要再|not\s+|don't\s+|do\s+not\s+|never\s+)$", prefix))


def detect_explicit_response_language(text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return None

    patterns: list[tuple[str, str]] = [
        (
            "Chinese",
            r"(?:请|麻烦|以后|后续|接下来|从现在起|之后|默认)?[^。！？\n]{0,16}"
            r"(?:用|使用|说|讲|以)[^。！？\n]{0,6}(?:中文|汉语|普通话)"
            r"[^。！？\n]{0,16}(?:交流|沟通|对话|回答|回复|聊天)",
        ),
        (
            "Chinese",
            r"(?:请|麻烦|以后|后续|接下来|从现在起|之后|默认)?[^。！？\n]{0,16}"
            r"(?:中文|汉语|普通话)[^。！？\n]{0,6}(?:回答|回复|交流|沟通|对话|聊天)",
        ),
        (
            "Chinese",
            r"(?:以后|后续|接下来|从现在起|之后|默认)[^。！？\n]{0,16}"
            r"(?:用|使用|说|讲|以)?[^。！？\n]{0,6}(?:中文|汉语|普通话)",
        ),
        (
            "English",
            r"(?:请|麻烦|以后|后续|接下来|从现在起|之后|默认)?[^。！？\n]{0,16}"
            r"(?:用|使用|说|讲|以)[^。！？\n]{0,6}(?:英文|英语|English|english)"
            r"[^。！？\n]{0,16}(?:交流|沟通|对话|回答|回复|聊天)",
        ),
        (
            "English",
            r"(?:请|麻烦|以后|后续|接下来|从现在起|之后|默认)?[^。！？\n]{0,16}"
            r"(?:英文|英语|English|english)[^。！？\n]{0,6}(?:回答|回复|交流|沟通|对话|聊天)",
        ),
        (
            "English",
            r"(?:以后|后续|接下来|从现在起|之后|默认)[^。！？\n]{0,16}"
            r"(?:用|使用|说|讲|以)?[^。！？\n]{0,6}(?:英文|英语|English|english)",
        ),
        (
            "English",
            r"\b(?:please\s+)?(?:communicate|reply|respond|speak|talk|chat)\s+"
            r"(?:with\s+me\s+)?in\s+English\b",
        ),
        (
            "English",
            r"\b(?:from now on|going forward|after this|for future responses|by default),?\s+"
            r"(?:please\s+)?(?:use|speak|reply|respond|communicate in)\s+English\b",
        ),
        (
            "English",
            r"\b(?:please\s+)?(?:use|switch to)\s+English\b"
            r"(?=\s*(?:[.\n!?]|$)|[^.\n!?]*(?:with me|from now on|going forward|by default|for future responses|to communicate|for our chat))",
        ),
        (
            "Chinese",
            r"\b(?:please\s+)?(?:communicate|reply|respond|speak|talk|chat)\s+"
            r"(?:with\s+me\s+)?in\s+(?:Chinese|Mandarin)\b",
        ),
        (
            "Chinese",
            r"\b(?:from now on|going forward|after this|for future responses|by default),?\s+"
            r"(?:please\s+)?(?:use|speak|reply|respond|communicate in)\s+(?:Chinese|Mandarin)\b",
        ),
        (
            "Chinese",
            r"\b(?:please\s+)?(?:use|switch to)\s+(?:Chinese|Mandarin)\b"
            r"(?=\s*(?:[.\n!?]|$)|[^.\n!?]*(?:with me|from now on|going forward|by default|for future responses|to communicate|for our chat))",
        ),
    ]

    matches: list[tuple[int, str]] = []
    for language, pattern in patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            if not _is_negated_language_match(normalized, match.start()):
                matches.append((match.start(), language))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def language_instruction_for(initial_language: str | None, explicit_language: str | None = None) -> str:
    if explicit_language == "English":
        return (
            "Language instruction: the user explicitly requested English. "
            "All future replies must be in English unless the user explicitly requests another language."
        )
    if explicit_language == "Chinese":
        return "语言指令：用户已明确要求使用中文。后续所有回复必须使用中文，除非用户再次明确指定其他语言。"
    if initial_language == "English":
        return (
            "Language instruction: the user's initial input language is English. "
            "Reply in English by default unless the user explicitly asks for another language."
        )
    if initial_language == "Chinese":
        return "语言指令：用户最初输入语言为中文。默认使用中文回答，除非用户明确要求其他语言。"
    return ""


def _locale_family(locale: str | None) -> str:
    normalized = (locale or "").strip().lower()
    if normalized.startswith("en"):
        return "en"
    return "zh"


def compression_progress_text(locale: str | None, key: str) -> str:
    family = _locale_family(locale)
    texts = {
        "zh": {
            "conversation_graph": "正在生成对话DAG...",
            "context_compression": "正在压缩上下文...",
        },
        "en": {
            "conversation_graph": "Building conversation graph...",
            "context_compression": "Compressing context...",
        },
    }
    return texts[family].get(key, texts[family]["context_compression"])


def trim_messages(messages: list[dict]) -> list[dict]:
    if len(messages) <= 1:
        return messages
    keep = 1 + MAX_ROUNDS * 2
    return [messages[0], *messages[-(keep - 1) :]]


def _message_has_image(message: dict) -> bool:
    content = message.get("content", [])
    if not isinstance(content, list):
        return False
    return any(isinstance(part, dict) and part.get("type") == "image" for part in content)


def _message_text(message: dict) -> str:
    content = message.get("content", [])
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            parts.append(str(part.get("text", "")))
        elif part.get("type") == "image":
            parts.append("[image]")
    return "\n".join(part for part in parts if part)


def _estimate_text_tokens(text: str) -> int:
    cjk_chars = 0
    other_chars = 0
    spaces = 0
    for char in text:
        code = ord(char)
        if char.isspace():
            spaces += 1
        elif (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x3040 <= code <= 0x30FF
            or 0xAC00 <= code <= 0xD7AF
        ):
            cjk_chars += 1
        else:
            other_chars += 1

    estimated = int(round(cjk_chars * 1.05 + other_chars / 3.7 + spaces / 10.0))
    return max(1, estimated)


class _TerminalStatusReporter:
    def __init__(self) -> None:
        self._last_key: tuple[int, str] | None = None

    def reset(self) -> None:
        self._last_key = None

    def emit(self, progress_ratio: float, status: str) -> None:
        del progress_ratio, status


def _quiet_stream_kwargs(callback: Callable[..., Any]) -> dict[str, bool]:
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return {"verbose": False}

    parameters = signature.parameters
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    quiet_kwargs: dict[str, bool] = {}
    for name, value in (
        ("verbose", False),
        ("show_progress", False),
        ("progress", False),
        ("progress_bar", False),
        ("disable_progress", True),
    ):
        if name in parameters or accepts_kwargs:
            quiet_kwargs[name] = value
    return quiet_kwargs


def _start_quiet_stream_generate(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    quiet_kwargs = _quiet_stream_kwargs(callback)
    try:
        return callback(*args, **kwargs, **quiet_kwargs)
    except TypeError as exc:
        message = str(exc)
        if quiet_kwargs and ("unexpected keyword" in message or "got an unexpected" in message):
            return callback(*args, **kwargs)
        raise


def _estimate_message_tokens(message: dict) -> int:
    return _estimate_text_tokens(_message_text(message))


def _message_to_item(message: dict, token_size: int | None = None, timestamp: float | None = None) -> MemoryItem:
    content = copy.deepcopy(message.get("content", []))
    if not isinstance(content, list):
        content = [{"type": "text", "text": str(content)}]
    return MemoryItem(
        role=str(message.get("role", "user")),
        content=content,
        timestamp=time.time() if timestamp is None else timestamp,
        token_size=max(1, token_size if token_size is not None else _estimate_message_tokens(message)),
        has_image=_message_has_image(message),
    )


def _read_model_profile(model_path: Path) -> ModelProfile:
    config_path = Path(model_path) / "config.json"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ModelProfile()

    text_config = data.get("text_config", {}) if isinstance(data, dict) else {}
    quantization = data.get("quantization", {}) if isinstance(data, dict) else {}
    vision_config = data.get("vision_config") if isinstance(data, dict) else None
    return ModelProfile(
        max_position_embeddings=text_config.get("max_position_embeddings"),
        num_hidden_layers=text_config.get("num_hidden_layers"),
        quant_bits=quantization.get("bits"),
        has_vision=vision_config is not None,
    )


class ContextCompressionScheduler:
    def __init__(self, system_message: dict, config: SchedulerConfig | None = None, model_path: Path = MODEL_PATH) -> None:
        self.config = config or scheduler_config_from_env()
        self.model_profile = _read_model_profile(model_path)
        self.hardware = HardwareProfile()
        self.telemetry = RuntimeTelemetry()
        self.state = ContextMemoryState(system_message=copy.deepcopy(system_message))
        self.completed_turns = 0
        self.compression_pool = self._new_compression_pool()
        self.last_structured_report: CompressionTurnReport | None = None
        self.phase = RuntimePhase.IDLE
        self._phase_lock = threading.RLock()
        self._generation_sem = threading.Semaphore(1)
        self._compression_sem = threading.Semaphore(1)
        self._compress_cancel_event = threading.Event()

    def clear(self) -> None:
        with self._phase_lock:
            self._compress_cancel_event.set()
            self.phase = RuntimePhase.CLEARING
            self.state.short_term.clear()
            self.state.mid_term.clear()
            self.state.long_term.clear()
            self.state.initial_user_language = None
            self.state.explicit_response_language = None
            self.state.revision += 1
            self.completed_turns = 0
            self.compression_pool = self._new_compression_pool()
            self.last_structured_report = None
            self._compress_cancel_event.clear()
            self.phase = RuntimePhase.IDLE

    def cancel_compression(self) -> None:
        self._compress_cancel_event.set()

    def mark_prefill_started(self, prepared_turn: PreparedTurn) -> None:
        del prepared_turn
        self._generation_sem.acquire()
        with self._phase_lock:
            self.phase = RuntimePhase.PREFILL

    def mark_decode_started(self, prepared_turn: PreparedTurn) -> None:
        del prepared_turn
        with self._phase_lock:
            self.phase = RuntimePhase.DECODE

    def mark_generation_finished(self) -> None:
        with self._phase_lock:
            self.phase = RuntimePhase.IDLE
        self._generation_sem.release()

    def observe_runtime_sample(self, stats: TurnStats) -> None:
        if stats.prompt_tps > 0:
            self.telemetry.prompt_tps_ema = self.telemetry.prompt_tps_ema * 0.65 + stats.prompt_tps * 0.35
        if stats.peak_memory > 0:
            self.telemetry.last_peak_memory_gb = stats.peak_memory
            self._observe_auto_k_candidate(stats.peak_memory)
            if stats.prompt_tokens > 0:
                observed_cost = max(DEFAULT_KV_COST_PER_TOKEN_GB, stats.peak_memory / max(1, stats.prompt_tokens) * 0.08)
                self.telemetry.kv_cost_per_token_gb = (
                    self.telemetry.kv_cost_per_token_gb * 0.75 + observed_cost * 0.25
                )

    def observe_compression_prefill_speed(self, prompt_tps: float) -> None:
        if prompt_tps > 0:
            self.telemetry.compression_prompt_tps_ema = (
                self.telemetry.compression_prompt_tps_ema * 0.65 + prompt_tps * 0.35
            )

    def _observe_auto_k_candidate(self, peak_memory_gb: float) -> None:
        if self.config.cage_budget_gb is not None or peak_memory_gb <= 0:
            return
        if peak_memory_gb > self.hardware.dynamic_ceiling_estimate_gb:
            self.hardware.dynamic_ceiling_estimate_gb = peak_memory_gb

    def _ensure_initial_user_language(self, text: str) -> None:
        if self.state.initial_user_language is not None:
            return
        language = detect_initial_user_language(text)
        if language is None:
            return
        self.state.initial_user_language = language
        self.state.revision += 1

    def _update_explicit_response_language(self, text: str) -> None:
        language = detect_explicit_response_language(text)
        if language is None or language == self.state.explicit_response_language:
            return
        self.state.explicit_response_language = language
        self.state.revision += 1

    def _language_instruction_message(self) -> dict | None:
        instruction = language_instruction_for(
            self.state.initial_user_language,
            self.state.explicit_response_language,
        )
        if not instruction:
            return None
        return build_user_message(instruction)

    def _base_instruction_messages(self) -> list[dict]:
        messages = [copy.deepcopy(self.state.system_message)]
        language_message = self._language_instruction_message()
        if language_message is not None:
            messages.append(language_message)
        return messages

    def prepare_turn(self, turn: UserTurn, build_prompt: PromptBuilder, estimate_tokens: TokenEstimator) -> PreparedTurn:
        with self._phase_lock:
            self.cancel_compression()
            self._compress_cancel_event.clear()
            self._ensure_initial_user_language(turn.text)
            self._update_explicit_response_language(turn.text)
            current_message = build_user_message(turn.text, turn.image_path)
            retrieval_context = ""
            retrieval_result = None
            if self.config.structured_compression_enabled:
                retrieval_result = self.compression_pool.retrieve_backtracking_context(turn.text)
                retrieval_context = retrieval_result.fused_context
            minimum_messages = self._base_instruction_messages()
            if self.config.structured_compression_enabled:
                structured_context = self.rendered_structured_context()
                if structured_context:
                    minimum_messages.append(build_text_message("assistant", f"[压缩上下文]\n{structured_context}"))
                if retrieval_context:
                    minimum_messages.append(build_text_message("assistant", retrieval_context))
            minimum_messages.append(copy.deepcopy(current_message))
            minimum_prompt = build_prompt(minimum_messages)
            minimum_tokens = estimate_tokens(minimum_messages, minimum_prompt)
            token_cap = self.resolve_prefill_context_cap(minimum_required_tokens=minimum_tokens)
            if minimum_tokens > token_cap:
                raise ContextBudgetError(
                    "prefill_context_cap_tokens is smaller than the minimum prompt required for this turn"
                )

            self._rebalance_for_token_budget(token_cap, reserved_tokens=minimum_tokens)
            messages, prompt_text, prompt_tokens = self._build_fit_prompt(
                current_message,
                retrieval_context,
                token_cap,
                build_prompt,
                estimate_tokens,
            )
            pressure = self.make_pressure_snapshot(prompt_tokens, token_cap, has_image=turn.image_path is not None)
            return PreparedTurn(
                turn=turn,
                revision=self.state.revision,
                prompt_messages=messages,
                prompt_text=prompt_text,
                prompt_tokens=prompt_tokens,
                pressure=pressure,
                retrieval_context=retrieval_context,
                retrieval_result=retrieval_result,
            )

    def finalize_turn(
        self,
        prepared_turn: PreparedTurn,
        answer: str,
        stats: TurnStats,
        build_prompt: PromptBuilder,
        estimate_tokens: TokenEstimator,
    ) -> CompressionOutcome:
        user_message = build_user_message(prepared_turn.turn.text, prepared_turn.turn.image_path)
        assistant_message = build_text_message("assistant", answer)
        self.observe_runtime_sample(stats)
        with self._phase_lock:
            self._remember_message(user_message)
            self._remember_message(assistant_message)
            self.completed_turns += 1
            if self.config.structured_compression_enabled:
                return CompressionOutcome()
            outcome = self.request_compression("post_turn", build_prompt, estimate_tokens)
            return outcome

    def request_structured_context_compression(
        self,
        raw_turn: str,
        llm_generate: Callable[[str, int], LLMResponse],
        semantic_dag: TurnSemanticDAG | None = None,
    ) -> CompressionOutcome:
        if not self.config.structured_compression_enabled:
            return CompressionOutcome()
        if not raw_turn.strip():
            return CompressionOutcome()
        if not self._generation_sem.acquire(blocking=False):
            return CompressionOutcome()
        acquired_compression = self._compression_sem.acquire(blocking=False)
        if not acquired_compression:
            self._generation_sem.release()
            return CompressionOutcome()

        try:
            with self._phase_lock:
                self.phase = RuntimePhase.COMPRESSING
                self._compress_cancel_event.clear()
                turn_id = self.completed_turns
            token_cap = self.resolve_prefill_context_cap(minimum_required_tokens=1)
            self.compression_pool.token_budget = token_cap
            self._sync_compression_pool_runtime_config()
            report = self.compression_pool.compress_turn_blocks(
                raw_turn,
                turn_id=turn_id,
                llm_generate=llm_generate,
                max_dirty_blocks=self.config.structured_max_dirty_blocks,
                target_tokens_per_block=self.config.structured_target_tokens_per_block,
                semantic_dag=semantic_dag,
            )
            with self._phase_lock:
                self.last_structured_report = report
                self._prune_after_structured_compression()
                self.state.revision += 1
            return CompressionOutcome(
                prompt_cache_invalidated=True,
                vision_cache_invalidated=self._layers_have_images(),
                compressed=True,
                structured_report=report,
            )
        finally:
            with self._phase_lock:
                self.phase = RuntimePhase.IDLE
            self._compression_sem.release()
            self._generation_sem.release()

    def index_structured_turn_dag(self, raw_turn: str) -> TurnSemanticDAG | None:
        if not self.config.structured_compression_enabled:
            return None
        if not raw_turn.strip():
            return None
        with self._phase_lock:
            turn_id = self.completed_turns
            return self.compression_pool.index_turn_memory(raw_turn, turn_id=turn_id)

    def request_structured_compression(
        self,
        raw_turn: str,
        llm_generate: Callable[[str, int], LLMResponse],
    ) -> CompressionOutcome:
        semantic_dag = self.index_structured_turn_dag(raw_turn)
        return self.request_structured_context_compression(raw_turn, llm_generate, semantic_dag=semantic_dag)

    def request_compression(
        self,
        reason: str,
        build_prompt: PromptBuilder,
        estimate_tokens: TokenEstimator,
    ) -> CompressionOutcome:
        del reason
        if not self._generation_sem.acquire(blocking=False):
            return CompressionOutcome()
        acquired_compression = self._compression_sem.acquire(blocking=False)
        if not acquired_compression:
            self._generation_sem.release()
            return CompressionOutcome()

        try:
            with self._phase_lock:
                self.phase = RuntimePhase.COMPRESSING
                self._compress_cancel_event.clear()
            token_cap = self.resolve_prefill_context_cap(minimum_required_tokens=1)
            before_revision = self.state.revision
            messages = self.export_model_messages()
            prompt_text = build_prompt(messages)
            prompt_tokens = estimate_tokens(messages, prompt_text)
            if prompt_tokens <= token_cap and not self._exceeds_effective_k(prompt_tokens, has_image=False):
                return CompressionOutcome()
            self._compress_until_fit(None, "", token_cap, build_prompt, estimate_tokens)
            compressed = self.state.revision != before_revision
            return CompressionOutcome(
                prompt_cache_invalidated=compressed,
                vision_cache_invalidated=compressed and self._layers_have_images(),
                compressed=compressed,
            )
        finally:
            with self._phase_lock:
                self.phase = RuntimePhase.IDLE
            self._compression_sem.release()
            self._generation_sem.release()

    def resolve_prefill_context_cap(self, minimum_required_tokens: int = 1) -> int:
        if self.config.prefill_context_cap_tokens is not None:
            # Treat a configured cap as a soft target. If the minimum runnable
            # prompt already exceeds it, elastically raise the effective cap
            # instead of failing the turn before compression can help.
            return max(1, self.config.prefill_context_cap_tokens, minimum_required_tokens)

        model_cap = self.model_profile.max_position_embeddings or 262144
        latency_cap = int(max(1.0, self.telemetry.prompt_tps_ema) * DEFAULT_AUTO_PREFILL_SECONDS)
        memory_cap = model_cap
        effective_k = self._effective_k_gb()
        if effective_k is not None and math.isfinite(effective_k):
            memory_cap = max(1, int(effective_k / max(self.telemetry.kv_cost_per_token_gb, 1e-9)))

        auto_cap = min(model_cap, max(MIN_AUTO_PREFILL_CONTEXT_TOKENS, latency_cap), memory_cap)
        return max(minimum_required_tokens, int(auto_cap))

    def make_pressure_snapshot(self, prompt_tokens: int, token_cap: int, has_image: bool = False) -> PressureSnapshot:
        projected_usage = self._project_usage_gb(prompt_tokens, has_image=has_image)
        effective_k = self._effective_k_gb()
        memory_ratio = projected_usage / effective_k if effective_k and math.isfinite(effective_k) and effective_k > 0 else 0.0
        token_ratio = prompt_tokens / max(1, token_cap)
        pressure_score = max(memory_ratio, token_ratio)
        urgency_score = 1.0 if pressure_score >= 1.0 else max(0.0, pressure_score - 0.8) / 0.2
        if pressure_score < 0.65:
            mode = "NONE"
        elif pressure_score < 0.82:
            mode = "LIGHT"
        elif pressure_score < 0.95:
            mode = "MEDIUM"
        elif pressure_score < 1.0:
            mode = "DEEP"
        else:
            mode = "EMERGENCY"
        debounce_ms = int(800 - min(1.0, urgency_score) * 700)
        return PressureSnapshot(
            projected_usage_gb=projected_usage,
            projected_prompt_tokens=prompt_tokens,
            pressure_score=pressure_score,
            urgency_score=urgency_score,
            compression_mode=mode,
            debounce_ms=max(0, debounce_ms),
            effective_prefill_context_cap_tokens=token_cap,
        )

    def export_model_messages(self) -> list[dict]:
        return self._messages_from_layers(current_message=None)

    def rendered_structured_context(self) -> str:
        if not self.config.structured_compression_enabled or self.last_structured_report is None:
            return ""
        self._sync_compression_pool_runtime_config()
        return self.compression_pool.render()

    def _effective_k_gb(self) -> float | None:
        if self.config.cage_budget_gb is not None:
            return max(0.0, self.config.cage_budget_gb)
        if self.hardware.dynamic_ceiling_estimate_gb > 0:
            return self.hardware.dynamic_ceiling_estimate_gb
        return None

    def _project_usage_gb(self, prompt_tokens: int, has_image: bool = False) -> float:
        usage = prompt_tokens * self.telemetry.kv_cost_per_token_gb
        if has_image:
            usage += self.telemetry.image_turn_penalty_gb
        return usage

    def _remember_message(self, message: dict) -> None:
        self.state.short_term.append(_message_to_item(message))
        self.state.revision += 1

    def _new_compression_pool(self) -> CompressionPool:
        pool = CompressionPool(
            blocks=CompressionPool.chat_blocks(),
            token_budget=self.resolve_prefill_context_cap(minimum_required_tokens=1),
            locale=self.config.locale,
        )
        self._sync_compression_pool_runtime_config(pool)
        return pool

    def _sync_compression_pool_runtime_config(self, pool: CompressionPool | None = None) -> None:
        target = self.compression_pool if pool is None else pool
        facts: list[str] = []
        effective_k = self._effective_k_gb()
        if effective_k is None:
            facts.append("K=auto")
        else:
            facts.append(f"K={effective_k:g}GB")
        facts.append(f"上下文上限={self.resolve_prefill_context_cap(minimum_required_tokens=1)} tokens")
        facts.append(f"prefill_step_size={PREFILL_STEP_SIZE}")
        facts.append(f"compression_prefill_step_size={COMPRESSION_PREFILL_STEP_SIZE}")
        compression_update_budget = max(
            256,
            int(round(self.config.compression_update_wait_seconds * max(1.0, self.telemetry.compression_prompt_tps_ema))),
        )
        target.configure_compression_update_budget(
            compression_update_budget,
            ramp_turns=DEFAULT_COMPRESSION_UPDATE_RAMP_TURNS,
        )
        facts.append(f"compression_update_wait={self.config.compression_update_wait_seconds:g}s")
        facts.append(f"compression_update_budget≈{compression_update_budget} tokens")
        facts.append(f"compression_update_ramp_turns={DEFAULT_COMPRESSION_UPDATE_RAMP_TURNS}")
        block = target.blocks.get("runtime_config")
        if block is not None:
            block.facts = facts
            block.protected_terms = facts.copy()
            block.updated_turn = self.completed_turns

    def _messages_from_layers(
        self,
        current_message: dict | None,
        retrieval_context_text: str = "",
    ) -> list[dict]:
        messages = self._base_instruction_messages()
        if self.config.structured_compression_enabled:
            structured_context = self.rendered_structured_context()
            if structured_context:
                messages.append(build_text_message("assistant", f"[压缩上下文]\n{structured_context}"))
            if retrieval_context_text:
                messages.append(build_text_message("assistant", retrieval_context_text))
            for item in sorted(self.state.short_term, key=lambda memory: memory.timestamp):
                messages.append(item.to_message())
            if current_message is not None:
                messages.append(copy.deepcopy(current_message))
            return messages

        for item in sorted(self.state.long_term, key=lambda memory: memory.timestamp):
            messages.append(item.to_message())
        for item in sorted(self.state.mid_term, key=lambda memory: memory.timestamp):
            messages.append(item.to_message())
        for item in sorted(self.state.short_term, key=lambda memory: memory.timestamp):
            messages.append(item.to_message())
        if retrieval_context_text:
            messages.append(build_text_message("assistant", retrieval_context_text))
        if current_message is not None:
            messages.append(copy.deepcopy(current_message))
        return messages

    def _prune_after_structured_compression(self) -> None:
        keep_items = max(0, self.config.structured_recent_turns) * 2
        if keep_items == 0:
            self.state.short_term.clear()
        elif len(self.state.short_term) > keep_items:
            self.state.short_term = self.state.short_term[-keep_items:]
        self.state.mid_term.clear()
        self.state.long_term.clear()

    def _build_fit_prompt(
        self,
        current_message: dict,
        retrieval_context_text: str,
        token_cap: int,
        build_prompt: PromptBuilder,
        estimate_tokens: TokenEstimator,
    ) -> tuple[list[dict], str, int]:
        messages = self._messages_from_layers(current_message, retrieval_context_text)
        prompt_text = build_prompt(messages)
        prompt_tokens = estimate_tokens(messages, prompt_text)
        if prompt_tokens <= token_cap and not self._exceeds_effective_k(prompt_tokens, _message_has_image(current_message)):
            return messages, prompt_text, prompt_tokens

        self._compress_until_fit(current_message, retrieval_context_text, token_cap, build_prompt, estimate_tokens)
        messages = self._messages_from_layers(current_message, retrieval_context_text)
        prompt_text = build_prompt(messages)
        prompt_tokens = estimate_tokens(messages, prompt_text)
        if prompt_tokens <= token_cap and not self._exceeds_effective_k(prompt_tokens, _message_has_image(current_message)):
            return messages, prompt_text, prompt_tokens

        messages, prompt_text, prompt_tokens = self._emergency_trimmed_messages(
            current_message,
            retrieval_context_text,
            token_cap,
            build_prompt,
            estimate_tokens,
        )
        if prompt_tokens > token_cap:
            # The configured prefill cap is a soft context target, not a hard
            # user-input length limit. If history has already been removed and
            # the minimum runnable prompt still exceeds it, allow the turn to
            # proceed unless the real memory budget rejects it below.
            if not self._exceeds_effective_k(prompt_tokens, _message_has_image(current_message)):
                return messages, prompt_text, prompt_tokens
            raise ContextBudgetError("Unable to fit prompt inside K dynamic memory budget")
        if self._exceeds_effective_k(prompt_tokens, _message_has_image(current_message)):
            raise ContextBudgetError("Unable to fit prompt inside K dynamic memory budget")
        return messages, prompt_text, prompt_tokens

    def _compress_until_fit(
        self,
        current_message: dict | None,
        retrieval_context_text: str,
        token_cap: int,
        build_prompt: PromptBuilder,
        estimate_tokens: TokenEstimator,
    ) -> None:
        for _ in range(COMPRESSION_MAX_PASSES):
            if self._compress_cancel_event.is_set():
                return
            messages = self._messages_from_layers(current_message, retrieval_context_text)
            prompt_text = build_prompt(messages)
            prompt_tokens = estimate_tokens(messages, prompt_text)
            has_image = _message_has_image(current_message) if current_message is not None else False
            if prompt_tokens <= token_cap and not self._exceeds_effective_k(prompt_tokens, has_image):
                return
            if not self._compress_one_step():
                return

    def _rebalance_for_token_budget(self, token_cap: int, reserved_tokens: int) -> None:
        available = max(1, token_cap - reserved_tokens)
        short_budget = int(available * 0.45)
        mid_budget = int(available * 0.35)
        long_budget = max(1, available - short_budget - mid_budget)

        while self._layer_tokens(self.state.short_term) > short_budget and len(self.state.short_term) > MIN_RECENT_MEMORY_ITEMS:
            self.state.mid_term.append(self.state.short_term.pop(0))
            self.state.revision += 1
        while self._layer_tokens(self.state.mid_term) > mid_budget and self.state.mid_term:
            self._aggregate_mid_item(self.state.mid_term.pop(0))
        self._truncate_long_term(long_budget)

    def _compress_one_step(self) -> bool:
        if self.state.mid_term:
            self._aggregate_mid_item(self.state.mid_term.pop(0))
            return True
        if len(self.state.short_term) > MIN_RECENT_MEMORY_ITEMS:
            self.state.mid_term.append(self.state.short_term.pop(0))
            self.state.revision += 1
            return True
        if self.state.long_term:
            before = self._layer_tokens(self.state.long_term)
            self._truncate_long_term(max(1, before // 2))
            return self._layer_tokens(self.state.long_term) < before
        if self.state.short_term:
            self.state.short_term.pop(0)
            self.state.revision += 1
            return True
        return False

    def _aggregate_mid_item(self, item: MemoryItem) -> None:
        summary = self._summarize_item(item)
        if self.state.long_term:
            long_item = self.state.long_term[-1]
            existing = _message_text(long_item.to_message())
            merged_text = f"{existing}\n{summary}".strip()
            long_item.content = [{"type": "text", "text": merged_text}]
            long_item.token_size = _estimate_text_tokens(merged_text)
            long_item.timestamp = min(long_item.timestamp, item.timestamp)
            long_item.has_image = long_item.has_image or item.has_image
        else:
            message = build_text_message("assistant", f"[长期记忆]\n{summary}")
            self.state.long_term.append(_message_to_item(message, timestamp=item.timestamp))
        self.state.revision += 1

    def _summarize_item(self, item: MemoryItem) -> str:
        text = _message_text(item.to_message()).replace("\n", " ").strip()
        if len(text) > 240:
            text = text[:237].rstrip() + "..."
        role = item.role
        image_note = " [image]" if item.has_image else ""
        return f"- {role}{image_note}: {text}"

    def _truncate_long_term(self, token_budget: int) -> None:
        if not self.state.long_term:
            return
        combined = "\n".join(_message_text(item.to_message()) for item in self.state.long_term).strip()
        if _estimate_text_tokens(combined) <= token_budget:
            return
        max_chars = max(32, token_budget * 4)
        truncated = combined[-max_chars:].lstrip()
        self.state.long_term = [_message_to_item(build_text_message("assistant", f"[长期记忆]\n{truncated}"))]
        self.state.revision += 1

    def _emergency_trimmed_messages(
        self,
        current_message: dict,
        retrieval_context_text: str,
        token_cap: int,
        build_prompt: PromptBuilder,
        estimate_tokens: TokenEstimator,
    ) -> tuple[list[dict], str, int]:
        base_messages = self._base_instruction_messages()
        selected: list[dict] = [*base_messages]
        if retrieval_context_text:
            selected.append(build_text_message("assistant", retrieval_context_text))
        selected.append(copy.deepcopy(current_message))
        prompt_text = build_prompt(selected)
        prompt_tokens = estimate_tokens(selected, prompt_text)
        if prompt_tokens > token_cap:
            return selected, prompt_text, prompt_tokens

        candidates = []
        for item in self.state.long_term + self.state.mid_term + self.state.short_term:
            candidates.append(item)
        candidates.sort(key=lambda memory: memory.timestamp, reverse=True)

        for item in candidates:
            trial = [*base_messages, item.to_message(), *selected[len(base_messages) :]]
            trial_prompt = build_prompt(trial)
            trial_tokens = estimate_tokens(trial, trial_prompt)
            if trial_tokens <= token_cap and not self._exceeds_effective_k(trial_tokens, _message_has_image(current_message)):
                selected = trial
                prompt_text = trial_prompt
                prompt_tokens = trial_tokens
        return selected, prompt_text, prompt_tokens

    def _exceeds_effective_k(self, prompt_tokens: int, has_image: bool) -> bool:
        effective_k = self._effective_k_gb()
        if effective_k is None or not math.isfinite(effective_k):
            return False
        return self._project_usage_gb(prompt_tokens, has_image=has_image) > effective_k

    def _layers_have_images(self) -> bool:
        return any(item.has_image for item in self.state.short_term + self.state.mid_term + self.state.long_term)

    @staticmethod
    def _layer_tokens(layer: list[MemoryItem]) -> int:
        return sum(max(1, item.token_size) for item in layer)


class ChatSession:
    def __init__(self, model_path: Path | str | None = None, scheduler_config: SchedulerConfig | None = None):
        configured_model_path = model_path
        if configured_model_path is None:
            configured_value = _get_config_value("LLM_MODEL_PATH")
            configured_model_path = Path(configured_value).expanduser() if configured_value else MODEL_PATH
            if not Path(configured_model_path).is_absolute():
                configured_model_path = Path(__file__).resolve().parent / configured_model_path
        self.model_path = Path(configured_model_path)
        self.model = None
        self.processor = None
        self.prompt_cache_state = None
        self.vision_cache = None
        self._load_jang_vlm_model = None
        self._vlm_stream_generate = None
        self._prompt_cache_state_cls = None
        self._vision_feature_cache_cls = None
        self._prompt_tps_ema = DEFAULT_PROMPT_TPS
        self._system_message = build_user_message(SYSTEM_PROMPT)
        self.scheduler = ContextCompressionScheduler(self._system_message, scheduler_config, self.model_path)
        self.messages = [copy.deepcopy(self._system_message)]
        self._prepared_turn: PreparedTurn | None = None
        self._terminal_status = _TerminalStatusReporter()
        self.trace_prefill = _parse_bool(_get_config_value("LLM_TRACE_PREFILL"), True)
        self.model_display_name = detect_model_display_name(self.model_path)

    def _ensure_runtime(self) -> None:
        if self._load_jang_vlm_model is not None:
            return
        from jang_tools.loader import load_jang_vlm_model
        from mlx_vlm import PromptCacheState, VisionFeatureCache
        from mlx_vlm import stream_generate as vlm_stream_generate

        self._load_jang_vlm_model = load_jang_vlm_model
        self._vlm_stream_generate = vlm_stream_generate
        self._prompt_cache_state_cls = PromptCacheState
        self._vision_feature_cache_cls = VisionFeatureCache
        self.prompt_cache_state = PromptCacheState()
        self.vision_cache = VisionFeatureCache(max_size=VISION_CACHE_SIZE)

    def load(self) -> None:
        self._ensure_runtime()
        if self.model is not None and self.processor is not None:
            return
        self.model, self.processor = self._load_jang_vlm_model(self.model_path)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def clear(self) -> None:
        self.scheduler.clear()
        self.messages = [copy.deepcopy(self._system_message)]
        self._prepared_turn = None
        self._reset_prompt_cache_state()
        if self.vision_cache is not None:
            self.vision_cache.clear()

    def _reset_prompt_cache_state(self) -> None:
        if self._prompt_cache_state_cls is not None:
            self.prompt_cache_state = self._prompt_cache_state_cls()
        else:
            self.prompt_cache_state = None

    def _messages_for_turn(self, turn: UserTurn) -> list[dict]:
        prepared = self._prepare_turn(turn)
        return copy.deepcopy(prepared.prompt_messages)

    def _build_prompt(self, messages: list[dict] | None = None) -> str:
        prompt_messages = messages if messages is not None else self.messages
        return self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _estimate_prompt_tokens(self, messages: list[dict], prompt_text: str) -> int:
        try:
            tokenized = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if isinstance(tokenized, dict):
                input_ids = tokenized.get("input_ids")
                if isinstance(input_ids, list):
                    if input_ids and isinstance(input_ids[0], list):
                        return max(1, len(input_ids[0]))
                    return max(1, len(input_ids))
            if isinstance(tokenized, list):
                if tokenized and isinstance(tokenized[0], list):
                    return max(1, len(tokenized[0]))
                return max(1, len(tokenized))
            if hasattr(tokenized, "shape"):
                shape = getattr(tokenized, "shape")
                if shape:
                    return max(1, int(shape[-1]))
            if hasattr(tokenized, "__len__"):
                return max(1, len(tokenized))
        except Exception:
            pass

        return _estimate_text_tokens(prompt_text)

    def _run_structured_compression_prompt(self, prompt: str, max_tokens: int) -> LLMResponse:
        messages = [build_user_message(prompt)]
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = self._estimate_prompt_tokens(messages, prompt_text)
        prompt_cache_state = self._prompt_cache_state_cls() if self._prompt_cache_state_cls is not None else None
        vision_cache = self._vision_feature_cache_cls(max_size=VISION_CACHE_SIZE) if self._vision_feature_cache_cls is not None else None

        chunks: list[str] = []
        observed_peak_memory = 0.0
        prompt_tps = 0.0
        generation_tps = 0.0
        generation_tokens = 0
        if self.trace_prefill:
            self._terminal_status.emit(0.0, "Context compression update started")
        start = time.perf_counter()
        stream = _start_quiet_stream_generate(
            self._vlm_stream_generate,
            self.model,
            self.processor,
            prompt=prompt_text,
            image=None,
            max_tokens=max_tokens,
            prefill_step_size=COMPRESSION_PREFILL_STEP_SIZE,
            prompt_cache_state=prompt_cache_state,
            vision_cache=vision_cache,
        )
        while True:
            try:
                chunk = next(stream)
            except StopIteration:
                break
            observed_peak_memory = max(observed_peak_memory, float(getattr(chunk, "peak_memory", 0.0)))
            prompt_tps = max(prompt_tps, float(getattr(chunk, "prompt_tps", 0.0)))
            generation_tps = max(generation_tps, float(getattr(chunk, "generation_tps", 0.0)))
            generation_tokens = max(generation_tokens, int(getattr(chunk, "generation_tokens", 0)))
            text = getattr(chunk, "text", "")
            if self.trace_prefill and not text:
                chunk_prompt_tokens = int(getattr(chunk, "prompt_tokens", 0) or 0)
                if chunk_prompt_tokens > 0 and prompt_tokens > 0:
                    self._terminal_status.emit(chunk_prompt_tokens / prompt_tokens, "Context compression update running")
            if text:
                chunks.append(text)

        wall_seconds = time.perf_counter() - start
        prefill_seconds = prompt_tokens / prompt_tps if prompt_tps > 0 else 0.0
        estimated_decode_seconds = generation_tokens / generation_tps if generation_tps > 0 else 0.0
        decode_seconds = max(estimated_decode_seconds, max(0.0, wall_seconds - prefill_seconds))
        if self.trace_prefill:
            self._terminal_status.emit(1.0, "Context compression update complete")
        self.scheduler.observe_runtime_sample(
            TurnStats(
                prompt_tokens=prompt_tokens,
                generation_tokens=generation_tokens,
                total_tokens=prompt_tokens + generation_tokens,
                prompt_tps=prompt_tps,
                generation_tps=generation_tps,
                peak_memory=observed_peak_memory,
            )
        )
        self.scheduler.observe_compression_prefill_speed(prompt_tps)
        return LLMResponse(
            text="".join(chunks).strip(),
            stats=LLMRunStats(
                input_tokens=prompt_tokens,
                output_tokens=generation_tokens,
                wall_seconds=wall_seconds,
                prefill_seconds=prefill_seconds,
                decode_seconds=decode_seconds,
                peak_memory_gb=observed_peak_memory,
            ),
        )

    def _raw_turn_for_structured_compression(self, prepared: PreparedTurn, answer: str) -> str:
        parts = [f"用户：{prepared.turn.text.strip()}"]
        if prepared.turn.image_path:
            parts.append(f"用户图片：{prepared.turn.image_path}")
        parts.append(f"助手：{answer.strip()}")
        return "\n".join(part for part in parts if part.strip())

    def _prepare_turn(self, turn: UserTurn) -> PreparedTurn:
        if self._prepared_turn is not None and self._prepared_turn.turn == turn:
            return self._prepared_turn
        prepared = self.scheduler.prepare_turn(turn, self._build_prompt, self._estimate_prompt_tokens)
        self._prepared_turn = prepared
        return prepared

    def estimate_prefill(self, turn: UserTurn) -> PrefillEstimate:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")
        prepared = self._prepare_turn(turn)
        prompt_tps = max(1.0, self._prompt_tps_ema)
        seconds = prepared.prompt_tokens / prompt_tps
        seconds = max(MIN_PREFILL_ESTIMATE_SECONDS, min(MAX_PREFILL_ESTIMATE_SECONDS, seconds))
        return PrefillEstimate(seconds=seconds, prompt_tokens=prepared.prompt_tokens)

    def should_show_backtracking_status(self, turn: UserTurn) -> bool:
        if not self.scheduler.config.structured_compression_enabled:
            return False
        if not turn.text.strip():
            return False
        return self.scheduler.compression_pool.backtracking_decision(turn.text).triggered

    def stream_turn(self, turn: UserTurn) -> Generator[PrefillProgress | CompressionProgress | TurnChunk, None, TurnResult]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        images = [turn.image_path] if turn.image_path else None
        if images is not None:
            # Hybrid prompt caches can desync when a new image token sequence is appended.
            self._reset_prompt_cache_state()

        prepared = self._prepare_turn(turn)
        self.messages = copy.deepcopy(prepared.prompt_messages)
        prompt_text = prepared.prompt_text
        estimated_prompt_tokens = prepared.prompt_tokens

        chunks: list[str] = []
        last_stats = TurnStats()
        observed_peak_memory = 0.0
        decode_started = False

        if self.trace_prefill:
            self._terminal_status.reset()
            if prepared.retrieval_result is not None and prepared.retrieval_result.decision.triggered:
                self._terminal_status.emit(1.0, "Backtracking retrieval ready")
            self._terminal_status.emit(0.0, "Main prefill started")
        self.scheduler.mark_prefill_started(prepared)
        try:
            stream = _start_quiet_stream_generate(
                self._vlm_stream_generate,
                self.model,
                self.processor,
                prompt=prompt_text,
                image=images,
                max_tokens=MAX_TOKENS,
                prefill_step_size=PREFILL_STEP_SIZE,
                prompt_cache_state=self.prompt_cache_state,
                vision_cache=self.vision_cache,
            )
            while True:
                try:
                    chunk = next(stream)
                except StopIteration:
                    break
                observed_peak_memory = max(observed_peak_memory, chunk.peak_memory)
                last_stats = TurnStats(
                    prompt_tokens=chunk.prompt_tokens,
                    generation_tokens=chunk.generation_tokens,
                    total_tokens=chunk.total_tokens,
                    prompt_tps=chunk.prompt_tps,
                    generation_tps=chunk.generation_tps,
                    peak_memory=observed_peak_memory,
                )
                if not chunk.text:
                    if chunk.prompt_tokens > 0 or chunk.prompt_tps > 0:
                        if self.trace_prefill and estimated_prompt_tokens > 0:
                            self._terminal_status.emit(
                                chunk.prompt_tokens / estimated_prompt_tokens,
                                "Main prefill running",
                            )
                        yield PrefillProgress(
                            prompt_tokens=max(0, min(chunk.prompt_tokens, estimated_prompt_tokens)),
                            total_prompt_tokens=estimated_prompt_tokens,
                            prompt_tps=chunk.prompt_tps,
                        )
                    continue
                if not decode_started:
                    self.scheduler.mark_decode_started(prepared)
                    decode_started = True
                    if self.trace_prefill:
                        self._terminal_status.emit(1.0, "Response generation started")
                text = chunk.text
                chunks.append(text)
                yield TurnChunk(text=text, stats=last_stats)
        except Exception:
            self.scheduler.mark_generation_finished()
            self._prepared_turn = None
            self._reset_prompt_cache_state()
            raise

        self.scheduler.mark_generation_finished()
        answer = "".join(chunks).strip()
        if last_stats.prompt_tps > 0:
            self._prompt_tps_ema = self._prompt_tps_ema * 0.65 + last_stats.prompt_tps * 0.35
        raw_structured_turn = self._raw_turn_for_structured_compression(prepared, answer)
        outcome = self.scheduler.finalize_turn(
            prepared,
            answer,
            last_stats,
            self._build_prompt,
            self._estimate_prompt_tokens,
        )
        semantic_dag = None
        if self.scheduler.config.structured_compression_enabled and raw_structured_turn.strip():
            yield CompressionProgress(
                message=compression_progress_text(self.scheduler.config.locale, "conversation_graph"),
                tone="active",
            )
            if self.trace_prefill:
                self._terminal_status.emit(0.0, "Conversation graph started")
            dag = self.scheduler.index_structured_turn_dag(raw_structured_turn)
            if self.trace_prefill:
                if dag is None:
                    self._terminal_status.emit(1.0, "Conversation graph skipped")
                else:
                    self._terminal_status.emit(1.0, "Conversation graph complete")
            semantic_dag = dag
            yield CompressionProgress(message=compression_progress_text(self.scheduler.config.locale, "context_compression"))
        structured_outcome = self.scheduler.request_structured_context_compression(
            raw_structured_turn,
            self._run_structured_compression_prompt,
            semantic_dag=semantic_dag,
        )
        if structured_outcome.compressed:
            outcome = structured_outcome
            if self.trace_prefill and structured_outcome.structured_report is not None:
                self._terminal_status.emit(1.0, "Context compression complete")
        if outcome.prompt_cache_invalidated:
            self._reset_prompt_cache_state()
        if outcome.vision_cache_invalidated and self.vision_cache is not None:
            self.vision_cache.clear()
        self.messages = self.scheduler.export_model_messages()
        self._prepared_turn = None
        return TurnResult(text=answer, stats=last_stats)

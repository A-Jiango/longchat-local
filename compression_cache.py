from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal


BlockStability = Literal["locked", "stable", "volatile"]
CHAT_TOPIC_RECENT_TURNS = 10
CHAT_TOPIC_FACTS_PER_TURN = 4
CHAT_TOPIC_FACT_MAX_CHARS = 88
CHAT_TOPIC_ARCHIVE_MAX_CHARS = 180
RAW_MEMORY_CHUNK_MAX_LINES = 3
RAW_MEMORY_CHUNK_MAX_CHARS = 220
BACKTRACK_RECENT_DAG_TURNS = 5
BACKTRACK_TOP_K_NODES = 3
BACKTRACK_MAX_HITS = 3
BACKTRACK_EXPAND_LINES = 2
BACKTRACK_MAX_SEGMENT_TOKENS = 180
BACKTRACK_CONTEXT_TOKEN_BUDGET = 560
BACKTRACK_MAX_NEIGHBOR_NODES = 4
DAG_MATERIAL_RECENT_FULL_TURNS = 1
DAG_MATERIAL_RECENT_MAX_NODES = 12
DAG_MATERIAL_RECENT_MAX_EDGES = 12
DAG_MATERIAL_AGED_MAX_NODES = 6
DAG_MATERIAL_TRACE_MAX_NODES = 1
DAG_MATERIAL_DIRECT_HISTORY_TURNS = 2
DAG_MATERIAL_HISTORY_SKETCH_MAX_ITEMS = 12
DAG_MATERIAL_SUMMARY_MAX_CHARS = 36
DAG_MATERIAL_AGED_SUMMARY_MAX_CHARS = 30
DAG_MATERIAL_TRACE_SUMMARY_MAX_CHARS = 22
DAG_MATERIAL_HISTORY_SKETCH_SUMMARY_MAX_CHARS = 28
DAG_MATERIAL_DECAY_RATE = 1.15
DAG_MATERIAL_TRACE_WEIGHT = 0.12
DAG_EVIDENCE_TARGET_RATIO = 1 / 3
DAG_EVIDENCE_MIN_TOKENS = 48
DAG_EVIDENCE_FULL_KEEP_TOKENS = 120
DAG_EVIDENCE_SOURCE_MAX_UNITS = 96
COMPRESSION_UPDATE_DEFAULT_WAIT_SECONDS = 20.0
COMPRESSION_UPDATE_DEFAULT_PREFILL_TPS = 120.0
COMPRESSION_UPDATE_RAMP_TURNS = 20
COMPRESSION_SIGNAL_EMA_ALPHA = 0.35
COMPRESSION_SIGNAL_MIN_MULTIPLIER = 0.55
COMPRESSION_SIGNAL_MAX_MULTIPLIER = 1.45
COMPRESSION_UPDATE_MIN_POSTERIOR_TOKENS = 64


@dataclass(frozen=True)
class SourceAnchor:
    turn_id: int
    chunk_id: str
    line_start: int
    line_end: int

    def payload(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "chunk_id": self.chunk_id,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }


@dataclass
class RawTextChunk:
    chunk_id: str
    turn_id: int
    ordinal: int
    line_start: int
    line_end: int
    text: str

    def payload(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "turn_id": self.turn_id,
            "ordinal": self.ordinal,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "text": self.text,
        }


@dataclass
class SemanticDAGEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0

    def payload(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": round(self.weight, 4),
        }


@dataclass
class SemanticDAGNode:
    node_id: str
    node_type: str
    text: str
    summary: str
    keywords: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    anchors: list[SourceAnchor] = field(default_factory=list)
    confidence: float = 1.0
    salience: float = 0.0

    def payload(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "text": self.text,
            "summary": self.summary,
            "keywords": self.keywords,
            "methods": self.methods,
            "objects": self.objects,
            "attributes": self.attributes,
            "anchors": [anchor.payload() for anchor in self.anchors],
            "confidence": round(self.confidence, 4),
            "salience": round(self.salience, 4),
        }


@dataclass
class CompressedDAGNode:
    node_id: str
    node_type: str
    summary: str
    anchor_refs: list[SourceAnchor] = field(default_factory=list)

    def payload(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "summary": self.summary,
            "anchor_refs": [anchor.payload() for anchor in self.anchor_refs],
        }


@dataclass
class TurnSemanticDAG:
    turn_id: int
    raw_text: str
    lines: list[str] = field(default_factory=list)
    chunks: list[RawTextChunk] = field(default_factory=list)
    nodes: list[SemanticDAGNode] = field(default_factory=list)
    edges: list[SemanticDAGEdge] = field(default_factory=list)
    compressed_nodes: list[CompressedDAGNode] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def payload(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "lines": self.lines,
            "chunks": [chunk.payload() for chunk in self.chunks],
            "nodes": [node.payload() for node in self.nodes],
            "edges": [edge.payload() for edge in self.edges],
            "compressed_nodes": [node.payload() for node in self.compressed_nodes],
        }


@dataclass
class DAGCompressionMaterial:
    turn_id: int
    precision: str
    age: int = 0
    decay_weight: float = 1.0
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def payload(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "precision": self.precision,
            "age": self.age,
            "decay": round(self.decay_weight, 4),
            "nodes": self.nodes,
            "edges": self.edges,
        }


@dataclass
class BacktrackingDecision:
    triggered: bool
    reasons: list[str] = field(default_factory=list)

    def payload(self) -> dict:
        return {"triggered": self.triggered, "reasons": self.reasons}


@dataclass
class RetrievalCandidate:
    turn_id: int
    node_id: str
    node_type: str
    text: str
    score: float
    confidence: float
    matched_keywords: list[str] = field(default_factory=list)
    anchors: list[SourceAnchor] = field(default_factory=list)
    neighbor_node_ids: list[str] = field(default_factory=list)

    def payload(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "node_id": self.node_id,
            "node_type": self.node_type,
            "text": self.text,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "matched_keywords": self.matched_keywords,
            "anchors": [anchor.payload() for anchor in self.anchors],
            "neighbor_node_ids": self.neighbor_node_ids,
        }


@dataclass
class SaturatedSegment:
    turn_id: int
    chunk_ids: list[str]
    line_start: int
    line_end: int
    text: str
    confidence: float
    source_node_ids: list[str] = field(default_factory=list)

    def payload(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "chunk_ids": self.chunk_ids,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "source_node_ids": self.source_node_ids,
        }


@dataclass
class BacktrackingRetrievalResult:
    decision: BacktrackingDecision
    candidates: list[RetrievalCandidate] = field(default_factory=list)
    segments: list[SaturatedSegment] = field(default_factory=list)
    fused_context: str = ""

    @property
    def triggered(self) -> bool:
        return self.decision.triggered

    def payload(self) -> dict:
        return {
            "decision": self.decision.payload(),
            "candidates": [candidate.payload() for candidate in self.candidates],
            "segments": [segment.payload() for segment in self.segments],
            "fused_context": self.fused_context,
        }


def estimate_text_tokens(text: str) -> int:
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
    return max(1, int(round(cjk_chars * 1.05 + other_chars / 3.7 + spaces / 10.0)))


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_evidence_units(text: str) -> list[str]:
    units: list[str] = []
    for raw_line in re.split(r"[\n\r]+", text):
        line = raw_line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[。！？!?；;])\s*", line)
        for part in parts:
            normalized = _normalize_space(part)
            if normalized:
                units.append(normalized)
    return units


@dataclass
class LLMRunStats:
    input_tokens: int = 0
    output_tokens: int = 0
    wall_seconds: float = 0.0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    peak_memory_gb: float = 0.0


@dataclass
class LLMResponse:
    text: str
    stats: LLMRunStats = field(default_factory=LLMRunStats)


LLMGenerator = Callable[[str, int], LLMResponse]


@dataclass
class CompressionBlock:
    id: str
    title: str
    block_type: str
    facts: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    source_turns: list[int] = field(default_factory=list)
    updated_turn: int = 0
    stability: BlockStability = "stable"
    depends_on: list[str] = field(default_factory=list)
    supports: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    protected_terms: list[str] = field(default_factory=list)
    forbidden_terms: list[str] = field(default_factory=list)
    max_facts: int = 6
    max_risks: int = 3
    dirty: bool = False
    stale: bool = False

    def payload(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "block_type": self.block_type,
            "facts": self.facts,
            "risks": self.risks,
            "source_turns": self.source_turns,
            "updated_turn": self.updated_turn,
            "stability": self.stability,
            "depends_on": self.depends_on,
            "supports": self.supports,
        }

    @property
    def content_hash(self) -> str:
        return _stable_hash({"facts": self.facts, "risks": self.risks})


@dataclass
class CompressionUpdateReport:
    block_id: str
    status: str
    update_budget_tokens: int = 0
    evidence_tokens: int = 0
    raw_evidence_tokens: int = 0
    evidence_target_tokens: int = 0
    old_block_tokens: int = 0
    dag_material_tokens: int = 0
    model_calls: int = 0
    retry_calls: int = 0
    fallback_used: bool = False
    cache_hit: bool = False
    validation_errors: list[str] = field(default_factory=list)
    stats: LLMRunStats = field(default_factory=LLMRunStats)
    old_hash: str = ""
    new_hash: str = ""


@dataclass
class CompressionTurnReport:
    turn_id: int
    routed_blocks: list[str]
    updates: list[CompressionUpdateReport]
    rendered_tokens: int
    rendered_context: str

    @property
    def total_model_calls(self) -> int:
        return sum(update.model_calls for update in self.updates)

    @property
    def total_retry_calls(self) -> int:
        return sum(update.retry_calls for update in self.updates)

    @property
    def total_input_tokens(self) -> int:
        return sum(update.stats.input_tokens for update in self.updates)

    @property
    def total_output_tokens(self) -> int:
        return sum(update.stats.output_tokens for update in self.updates)

    @property
    def wall_seconds(self) -> float:
        return sum(update.stats.wall_seconds for update in self.updates)

    @property
    def prefill_seconds(self) -> float:
        return sum(update.stats.prefill_seconds for update in self.updates)

    @property
    def decode_seconds(self) -> float:
        return sum(update.stats.decode_seconds for update in self.updates)

    @property
    def peak_memory_gb(self) -> float:
        peaks = [update.stats.peak_memory_gb for update in self.updates]
        return max(peaks) if peaks else 0.0


@dataclass(frozen=True)
class LocaleResources:
    locale: str
    data: dict[str, Any]

    @classmethod
    def load(cls, locale: str = "zh-CN", base_dir: Path | None = None) -> "LocaleResources":
        locale_name = locale or "zh-CN"
        root = Path(__file__).resolve().with_name("locales") if base_dir is None else Path(base_dir)
        path = root / f"{locale_name}.json"
        if not path.exists() and locale_name != "zh-CN":
            path = root / "zh-CN.json"
            locale_name = "zh-CN"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        return cls(locale=locale_name, data=data)

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name, {})
        return value if isinstance(value, dict) else {}

    def list(self, section: str, key: str) -> list[str]:
        value = self.section(section).get(key, [])
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item)]

    def text(self, section: str, key: str, default: str = "") -> str:
        value = self.section(section).get(key, default)
        return str(value) if value is not None else default

    def nested_text(self, section: str, parent: str, key: str, default: str = "") -> str:
        value = self.section(section).get(parent, {})
        if not isinstance(value, dict):
            return default
        return str(value.get(key, default))


class EnglishNLPAdapter:
    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self.spacy_model = spacy_model or "en_core_web_sm"
        self._nlp = None
        self._stemmer = None
        self.backends: list[str] = []
        self._load_spacy()
        self._load_nltk()
        if not self.backends:
            self.backends.append("fallback")

    def _load_spacy(self) -> None:
        try:
            import spacy
        except Exception:
            return
        try:
            self._nlp = spacy.load(self.spacy_model, disable=["parser", "ner"])
            self.backends.append(f"spacy:{self.spacy_model}")
            return
        except Exception:
            pass
        try:
            self._nlp = spacy.blank("en")
            self.backends.append("spacy:blank-en")
        except Exception:
            self._nlp = None

    def _load_nltk(self) -> None:
        try:
            from nltk.stem import PorterStemmer
        except Exception:
            return
        try:
            self._stemmer = PorterStemmer()
            self.backends.append("nltk:porter")
        except Exception:
            self._stemmer = None

    @staticmethod
    def _regex_tokens(text: str) -> list[str]:
        return re.findall(r"\b[a-zA-Z][a-zA-Z'’]*\b", text.lower())

    @staticmethod
    def _clean_token(text: str) -> str:
        token = text.lower().strip("'’")
        return token if re.search(r"[a-z]", token) else ""

    @staticmethod
    def _clean_lemma(text: str) -> str:
        lemma = EnglishNLPAdapter._clean_token(text)
        if lemma == "-pron-":
            return ""
        return lemma

    def token_variants(self, text: str) -> list[set[str]]:
        if self._nlp is not None:
            raw_tokens = []
            for token in self._nlp(text):
                if token.is_space or token.is_punct:
                    continue
                raw = self._clean_token(token.text)
                if not raw:
                    continue
                variants = CompressionPool._word_variants(raw)
                lemma = self._clean_lemma(getattr(token, "lemma_", "") or "")
                if lemma:
                    variants.update(CompressionPool._word_variants(lemma))
                raw_tokens.append(self._stem_variants(variants))
            if raw_tokens:
                return raw_tokens
        return [self._stem_variants(CompressionPool._word_variants(token)) for token in self._regex_tokens(text)]

    def _stem_variants(self, variants: set[str]) -> set[str]:
        if self._stemmer is None:
            return variants
        stemmed = set(variants)
        for item in list(variants):
            try:
                stemmed.add(self._stemmer.stem(item))
            except Exception:
                continue
        return {item for item in stemmed if item}

    def term_matches(self, term: str, text: str) -> bool:
        term_tokens = self.token_variants(term)
        text_tokens = self.token_variants(text)
        if not term_tokens or not text_tokens:
            return False
        if len(term_tokens) == 1:
            return any(term_tokens[0] & token for token in text_tokens)
        if len(text_tokens) < len(term_tokens):
            return False
        for start in range(0, len(text_tokens) - len(term_tokens) + 1):
            window = text_tokens[start : start + len(term_tokens)]
            if all(term_variant & text_variant for term_variant, text_variant in zip(term_tokens, window)):
                return True
        return False

    def keyword_candidates(self, text: str) -> list[str]:
        candidates: list[str] = []
        if self._nlp is not None:
            for token in self._nlp(text):
                if token.is_space or token.is_punct:
                    continue
                raw = self._clean_token(token.text)
                if not raw:
                    continue
                lemma = self._clean_lemma(getattr(token, "lemma_", "") or "")
                candidates.append(lemma or raw)
        else:
            candidates.extend(self._regex_tokens(text))
        cleaned: list[str] = []
        for candidate in candidates:
            candidate = self._clean_token(candidate)
            if len(candidate) < 2:
                continue
            cleaned.append(candidate)
        return cleaned


class CompressionPool:
    policy_version = "dependency-graph-v1"
    prompt_version = "dirty-block-json-v4"

    def __init__(
        self,
        blocks: Iterable[CompressionBlock] | None = None,
        token_budget: int = 1000,
        locale: str = "zh-CN",
        locale_resources: LocaleResources | None = None,
    ) -> None:
        self.token_budget = token_budget
        self.locale = locale_resources or LocaleResources.load(locale)
        self._english_nlp = (
            EnglishNLPAdapter(self.locale.text("nlp", "spacy_model", "en_core_web_sm"))
            if self.locale.locale.lower().startswith("en")
            else None
        )
        self.compression_update_token_budget = int(
            round(COMPRESSION_UPDATE_DEFAULT_WAIT_SECONDS * COMPRESSION_UPDATE_DEFAULT_PREFILL_TPS)
        )
        self.compression_update_ramp_turns = COMPRESSION_UPDATE_RAMP_TURNS
        self._active_compression_update_budget_tokens = self.compression_update_token_budget
        self._compression_signal_token_ema = 0.0
        self._compression_signal_truncation_ema = 0.0
        self._compression_signal_dag_node_ema = 0.0
        self._last_compression_raw_evidence_tokens: int | None = None
        self._last_compression_dag_nodes: int | None = None
        self.blocks: dict[str, CompressionBlock] = {}
        self.block_cache: dict[str, CompressionBlock] = {}
        self.turn_dag_cache: list[TurnSemanticDAG] = []
        self.dag_material_cache: list[DAGCompressionMaterial] = []
        if blocks is None:
            blocks = self.default_blocks()
        for block in blocks:
            self.blocks[block.id] = copy.deepcopy(block)
        self._apply_locale_resources_to_blocks()

    def _apply_locale_resources_to_blocks(self) -> None:
        preference_block = self.blocks.get("user_preference")
        if preference_block is None:
            return
        preference_terms = [
            *self.locale.list("preference", "concrete_terms"),
            *self.locale.list("preference", "first_person_markers"),
            *self.locale.list("preference", "preference_terms"),
        ]
        preference_block.keywords = self._dedupe_keep_order([*preference_block.keywords, *preference_terms])

    def configure_compression_update_budget(self, token_budget: int, ramp_turns: int = COMPRESSION_UPDATE_RAMP_TURNS) -> None:
        self.compression_update_token_budget = max(256, int(token_budget))
        self.compression_update_ramp_turns = max(1, int(ramp_turns))
        self._active_compression_update_budget_tokens = self.compression_update_token_budget

    @staticmethod
    def _ema(previous: float, observed: float, alpha: float = COMPRESSION_SIGNAL_EMA_ALPHA) -> float:
        if previous <= 0:
            return float(observed)
        return previous * (1.0 - alpha) + float(observed) * alpha

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def _compression_update_linear_prior_budget(self, turn_id: int | None = None) -> int:
        if turn_id is None:
            progress = 1.0
        else:
            progress = self._clamp(turn_id / max(1, self.compression_update_ramp_turns), 0.20, 1.0)
        return max(1, int(round(self.compression_update_token_budget * progress)))

    def _compression_update_budget_signal_multiplier(
        self,
        raw_evidence_tokens: int,
        dag_node_count: int,
        prior_budget_tokens: int,
    ) -> float:
        current_tokens = max(1, int(raw_evidence_tokens))
        current_nodes = max(1, int(dag_node_count))
        token_baseline = self._compression_signal_token_ema or current_tokens
        node_baseline = self._compression_signal_dag_node_ema or current_nodes
        previous_tokens = self._last_compression_raw_evidence_tokens or current_tokens
        previous_nodes = self._last_compression_dag_nodes or current_nodes

        token_pressure = self._clamp(current_tokens / max(1.0, token_baseline), 0.4, 2.2) - 1.0
        token_growth = self._clamp(current_tokens / max(1, previous_tokens), 0.35, 2.4) - 1.0
        node_pressure = self._clamp(current_nodes / max(1.0, node_baseline), 0.5, 2.0) - 1.0
        node_growth = self._clamp(current_nodes / max(1, previous_nodes), 0.4, 2.4) - 1.0

        prior_evidence_target = self._dag_evidence_token_target_for_budget(current_tokens, prior_budget_tokens)
        truncation_ratio = max(0.0, 1.0 - prior_evidence_target / max(1, current_tokens))
        truncation_delta = truncation_ratio - self._compression_signal_truncation_ema

        multiplier = 1.0
        multiplier += token_pressure * 0.28
        multiplier += token_growth * 0.24
        multiplier += max(-0.25, truncation_delta) * 0.26
        multiplier += node_pressure * 0.10
        multiplier += node_growth * 0.12
        return self._clamp(
            multiplier,
            COMPRESSION_SIGNAL_MIN_MULTIPLIER,
            COMPRESSION_SIGNAL_MAX_MULTIPLIER,
        )

    def _compression_update_posterior_budget(
        self,
        turn_id: int,
        raw_evidence_tokens: int,
        semantic_dag: TurnSemanticDAG | None = None,
        *,
        update_state: bool = True,
    ) -> int:
        dag_node_count = len(semantic_dag.nodes) if semantic_dag is not None else 0
        prior = self._compression_update_linear_prior_budget(turn_id)
        multiplier = self._compression_update_budget_signal_multiplier(raw_evidence_tokens, dag_node_count, prior)
        posterior = max(
            COMPRESSION_UPDATE_MIN_POSTERIOR_TOKENS,
            int(round(prior * multiplier)),
        )
        posterior = min(self.compression_update_token_budget, posterior)

        if update_state:
            current_tokens = max(1, int(raw_evidence_tokens))
            current_nodes = max(1, int(dag_node_count))
            prior_evidence_target = self._dag_evidence_token_target_for_budget(current_tokens, prior)
            truncation_ratio = max(0.0, 1.0 - prior_evidence_target / max(1, current_tokens))
            self._compression_signal_token_ema = self._ema(self._compression_signal_token_ema, current_tokens)
            self._compression_signal_truncation_ema = self._ema(
                self._compression_signal_truncation_ema,
                truncation_ratio,
            )
            self._compression_signal_dag_node_ema = self._ema(self._compression_signal_dag_node_ema, current_nodes)
            self._last_compression_raw_evidence_tokens = current_tokens
            self._last_compression_dag_nodes = current_nodes
            self._active_compression_update_budget_tokens = posterior
        return posterior

    def _current_compression_update_budget_tokens(self) -> int:
        return max(1, int(self._active_compression_update_budget_tokens))

    def _compression_update_growth_tokens(self) -> int:
        return max(1, int(round(self._current_compression_update_budget_tokens() / max(1, self.compression_update_ramp_turns))))

    def _dynamic_evidence_max_tokens(self) -> int:
        budget = self._current_compression_update_budget_tokens()
        hard_cap = max(DAG_EVIDENCE_MIN_TOKENS, self.compression_update_token_budget // 4)
        return max(DAG_EVIDENCE_MIN_TOKENS, min(hard_cap, int(round(budget * 0.20))))

    def _dynamic_old_block_max_tokens(self) -> int:
        budget = self._current_compression_update_budget_tokens()
        hard_cap = max(128, self.compression_update_token_budget // 3)
        return max(128, min(hard_cap, int(round(budget * 0.14))))

    def _dynamic_recent_dag_node_limit(self) -> int:
        budget = self._current_compression_update_budget_tokens()
        return max(2, min(DAG_MATERIAL_RECENT_MAX_NODES, budget // 340))

    def _dynamic_recent_dag_edge_limit(self) -> int:
        budget = self._current_compression_update_budget_tokens()
        return max(1, min(DAG_MATERIAL_RECENT_MAX_EDGES, budget // 430))

    def _dynamic_aged_dag_node_limit(self) -> int:
        budget = self._current_compression_update_budget_tokens()
        return max(1, min(DAG_MATERIAL_AGED_MAX_NODES, budget // 900))

    def _dynamic_history_sketch_item_limit(self) -> int:
        budget = self._current_compression_update_budget_tokens()
        return max(2, min(DAG_MATERIAL_HISTORY_SKETCH_MAX_ITEMS, budget // 280))

    @staticmethod
    def default_blocks() -> list[CompressionBlock]:
        return [
            CompressionBlock(
                id="project_mainline",
                title="项目主线",
                block_type="main",
                stability="locked",
                facts=[
                    "目标：本地 LLM 桌面助手的增量上下文压缩",
                    "核心指标：首 token 时间、显存上限、主线保持",
                ],
                keywords=["项目主线", "目标", "本地 LLM", "桌面助手", "上下文压缩", "首 token", "显存", "主线保持"],
                protected_terms=["本地 LLM", "增量上下文压缩", "首 token", "显存", "主线"],
                forbidden_terms=["注意力权重", "语义相似度", "量化", "内存访问", "Transformer"],
                supports=["compression_architecture", "latency_strategy", "memory_strategy"],
                max_facts=4,
            ),
            CompressionBlock(
                id="runtime_config",
                title="运行配置",
                block_type="config",
                stability="locked",
                facts=["K=10GB", "上下文上限=1000 tokens", "prefill_step_size=256"],
                keywords=["K=", "10GB", "上下文上限", "1000", "prefill_step_size", "256", "配置"],
                protected_terms=["K=10GB", "1000", "prefill_step_size=256"],
                supports=["compression_architecture", "latency_strategy", "memory_strategy"],
                max_facts=5,
            ),
            CompressionBlock(
                id="compression_architecture",
                title="压缩架构",
                block_type="architecture",
                stability="stable",
                facts=[
                    "三层缓存池：canonical_pool、block_cache、rendered_context",
                    "每轮执行增头、aging、reweight、去尾、render",
                    "每轮只更新约 10%-20% dirty block，其余缓存复用",
                ],
                depends_on=["project_mainline", "runtime_config"],
                supports=["confirmed_decisions", "latency_strategy"],
                keywords=[
                    "canonical_pool",
                    "block_cache",
                    "rendered_context",
                    "dirty block",
                    "10%-20%",
                    "缓存池",
                    "增头",
                    "去尾",
                    "render",
                    "依赖图",
                    "dependency graph",
                    "stable block lock",
                    "evidence filter",
                    "retry budget",
                ],
                protected_terms=["canonical_pool", "block_cache", "rendered_context", "10%-20%", "去尾", "增头"],
                forbidden_terms=["注意力权重", "语义相似度", "量化", "内存访问", "矩阵乘法"],
                max_facts=7,
            ),
            CompressionBlock(
                id="confirmed_decisions",
                title="已确认决策",
                block_type="decision",
                stability="stable",
                facts=["默认并行压缩方案作废", "正式路径采用互斥执行", "压缩复用 generation_sem 和 compression_sem"],
                depends_on=["compression_architecture"],
                keywords=["作废", "正式路径", "互斥", "generation_sem", "compression_sem", "信号量", "默认并行", "并行压缩"],
                protected_terms=["互斥", "generation_sem", "compression_sem", "作废"],
                max_facts=6,
            ),
            CompressionBlock(
                id="risk_safety",
                title="风险与安全",
                block_type="risk",
                stability="stable",
                facts=[],
                risks=["需抵御提示词注入攻击", "需确保系统提示词不泄露"],
                depends_on=["project_mainline"],
                keywords=["注入", "提示词", "系统提示词", "泄露", "validator", "校验", "失败时", "兜底", "完整重试", "局部补丁"],
                protected_terms=["提示词注入", "系统提示词", "泄露", "完整重试", "兜底"],
                max_facts=4,
                max_risks=5,
            ),
            CompressionBlock(
                id="user_preference",
                title="用户偏好",
                block_type="preference",
                stability="volatile",
                facts=[],
                keywords=["喜欢", "不喜欢", "想吃", "不吃", "今晚", "以后", "清淡", "辣", "微辣", "饮食", "口味"],
                max_facts=6,
            ),
            CompressionBlock(
                id="recent_delta",
                title="最近变化",
                block_type="recent",
                stability="volatile",
                facts=[],
                keywords=[],
                max_facts=18,
            ),
        ]

    @staticmethod
    def chat_blocks() -> list[CompressionBlock]:
        return [
            CompressionBlock(
                id="conversation_summary",
                title="对话主题",
                block_type="topic",
                stability="volatile",
                facts=[],
                keywords=[],
                max_facts=22,
                max_risks=4,
            ),
            CompressionBlock(
                id="user_preference",
                title="用户偏好",
                block_type="preference",
                stability="volatile",
                facts=[],
                keywords=["喜欢", "不喜欢", "偏好", "想吃", "不吃", "今晚", "以后", "清淡", "辣", "微辣", "饮食", "口味"],
                max_facts=6,
            ),
            CompressionBlock(
                id="recent_delta",
                title="最近变化",
                block_type="recent",
                stability="volatile",
                facts=[],
                keywords=[],
                max_facts=18,
            ),
        ]

    @staticmethod
    def _authored_user_text(raw_turn: str) -> str:
        lines: list[str] = []
        saw_role = False
        for raw_line in raw_turn.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            user_match = re.match(r"^(?:用户|user)\s*[：:]\s*(.*)$", line, flags=re.I)
            if user_match:
                saw_role = True
                lines.append(user_match.group(1).strip())
                continue
            if re.match(r"^(?:助手|assistant)\s*[：:]", line, flags=re.I):
                saw_role = True
                continue
            if not saw_role:
                lines.append(line)
        return "\n".join(item for item in lines if item).strip()

    @staticmethod
    def _role_texts(raw_turn: str) -> tuple[list[str], list[str]]:
        user_parts: list[str] = []
        assistant_parts: list[str] = []
        current_role: str | None = None
        for raw_line in raw_turn.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            user_match = re.match(r"^(?:用户|user)\s*[：:]\s*(.*)$", line, flags=re.I)
            if user_match:
                current_role = "user"
                text = user_match.group(1).strip()
                if text:
                    user_parts.append(text)
                continue
            assistant_match = re.match(r"^(?:助手|assistant)\s*[：:]\s*(.*)$", line, flags=re.I)
            if assistant_match:
                current_role = "assistant"
                text = assistant_match.group(1).strip()
                if text:
                    assistant_parts.append(text)
                continue
            if current_role == "user":
                user_parts.append(line)
            elif current_role == "assistant":
                assistant_parts.append(line)
        return user_parts, assistant_parts

    @staticmethod
    def _short_dialogue_excerpt(text: str, max_chars: int) -> str:
        cleaned = CompressionPool._clean_fact(text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip("，,；;。 ")
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 1].rstrip("，,；;。 ") + "…"

    def _is_question_like(self, text: str) -> bool:
        normalized = _normalize_space(text)
        if not normalized:
            return False
        if any(marker and marker in normalized for marker in self.locale.list("question", "markers")):
            return True
        lowered = normalized.lower()
        return any(marker and marker.lower() in lowered for marker in self.locale.list("question", "case_insensitive_markers"))

    def _conversation_turn_meta(self, raw_turn: str) -> str:
        user_parts, assistant_parts = CompressionPool._role_texts(raw_turn)
        user_text = _normalize_space(" ".join(user_parts))
        if not user_text:
            return ""
        if self._conversation_memory_query_intent(user_text):
            verb_key = "memory_query"
        elif self._is_question_like(user_text):
            verb_key = "question"
        else:
            verb_key = "request"
        verb = self.locale.nested_text("dialogue_meta", "verbs", verb_key, verb_key)
        user_excerpt = CompressionPool._short_dialogue_excerpt(user_text, 36)
        template = self.locale.text("dialogue_meta", "template", '本轮对话：用户{verb}“{excerpt}”')
        return template.format(verb=verb, excerpt=user_excerpt)

    def _is_dialogue_meta_evidence(self, text: str) -> bool:
        normalized = _normalize_space(text)
        return any(prefix and normalized.startswith(prefix) for prefix in self.locale.list("dialogue_meta", "prefixes"))

    def _dialogue_meta_user_excerpt(self, text: str) -> str:
        normalized = _normalize_space(text)
        for pattern in self.locale.list("dialogue_meta", "user_excerpt_patterns"):
            try:
                match = re.search(pattern, normalized, flags=re.I)
            except re.error:
                continue
            if match:
                return match.group(1).strip()
        match = re.search(r"[“\"]([^”\"]+)[”\"]", normalized)
        if not match:
            return ""
        return match.group(1).strip()

    def _conversation_memory_query_intent(self, user_text: str) -> str:
        normalized = _normalize_space(user_text)
        if not normalized:
            return ""
        lowered = normalized.lower()
        markers = self.locale.list("memory_query", "markers")
        if not any(marker and (marker in normalized or marker.lower() in lowered) for marker in markers):
            return ""
        question = normalized[:80].rstrip("。；;")
        template = self.locale.text("memory_query", "intent_template", "用户在追问先前聊天记录：{question}")
        return template.format(question=question)

    def _has_user_preference_signal(self, text: str) -> bool:
        normalized = _normalize_space(text)
        if not normalized:
            return False
        concrete_terms = self.locale.list("preference", "concrete_terms")
        if any(self._term_matches(term, normalized) for term in concrete_terms):
            return True
        first_person_markers = self.locale.list("preference", "first_person_markers")
        preference_terms = self.locale.list("preference", "preference_terms")
        return any(self._term_matches(marker, normalized) for marker in first_person_markers) and any(
            self._term_matches(term, normalized) for term in preference_terms
        )

    def route_dirty_blocks(self, raw_turn: str, max_blocks: int = 2) -> list[str]:
        scores: list[tuple[int, int, str]] = []
        normalized = raw_turn
        user_text = self._authored_user_text(raw_turn)
        preference_signal = self._has_user_preference_signal(user_text)
        for index, block in enumerate(self.blocks.values()):
            if block.id == "recent_delta":
                continue
            if block.id == "user_preference" and not preference_signal:
                continue
            if block.stability == "locked" and not self._locked_update_allowed(block, [raw_turn]):
                continue
            score = 0
            search_text = user_text if block.id == "user_preference" else normalized
            for keyword in block.keywords:
                if keyword and self._term_matches(keyword, search_text):
                    score += 3 if block.stability == "locked" else 2
            for term in block.protected_terms:
                if term and self._term_matches(term, search_text):
                    score += 2
            if preference_signal and block.id == "user_preference":
                score += 10
            if preference_signal and block.id == "project_mainline":
                score = max(0, score - 10)
            if score > 0:
                scores.append((score, -index, block.id))

        if not scores:
            if "conversation_summary" in self.blocks:
                return ["conversation_summary"]
            return ["recent_delta"]

        scores.sort(reverse=True)
        routed: list[str] = []
        for _, _, block_id in scores:
            if block_id not in routed:
                routed.append(block_id)
            if len(routed) >= max_blocks:
                break
        return routed

    def filter_evidence(
        self,
        raw_turn: str,
        block_id: str,
        max_units: int = 8,
    ) -> list[str]:
        block = self.blocks[block_id]
        units = split_evidence_units(raw_turn)
        if block.id == "recent_delta":
            return units[-max_units:]
        if block.id == "conversation_summary":
            memory_query_intent = self._conversation_memory_query_intent(self._authored_user_text(raw_turn))
            turn_meta = self._conversation_turn_meta(raw_turn)
            evidence = [
                unit
                for unit in units
                if not self._is_low_value_fact(block, self._clean_fact(unit))
                and not (turn_meta and re.match(r"^(?:用户|user)\s*[：:]", unit.strip(), flags=re.I))
                and not (
                    memory_query_intent
                    and re.match(r"^(?:用户|user)\s*[：:]", unit.strip(), flags=re.I)
                    and self._conversation_memory_query_intent(self._clean_fact(unit))
                )
            ]
            if turn_meta:
                evidence.insert(0, turn_meta)
            if memory_query_intent:
                evidence.insert(0, memory_query_intent)
            return self._dedupe_keep_order(evidence)[: max(max_units, 24)]
        if block.id == "user_preference":
            units = split_evidence_units(self._authored_user_text(raw_turn))

        evidence: list[str] = []
        preference_block = self.blocks.get("user_preference")
        preference_keywords = preference_block.keywords if preference_block is not None else []
        for unit in units:
            cleaned_unit = self._clean_fact(unit)
            if self._is_low_value_fact(block, cleaned_unit):
                continue
            if block.id == "project_mainline" and any(
                self._term_matches(keyword, unit) for keyword in preference_keywords if keyword
            ):
                continue
            if block.id == "user_preference" and not self._has_user_preference_signal(cleaned_unit):
                continue
            has_keyword = any(self._term_matches(keyword, unit) for keyword in block.keywords if keyword)
            has_protected = any(self._term_matches(term, unit) for term in block.protected_terms if term)
            has_forbidden = any(self._term_matches(term, unit) for term in block.forbidden_terms if term)
            if has_forbidden and not (has_keyword or has_protected):
                continue
            if has_keyword or has_protected:
                evidence.append(unit)
        return self._dedupe_keep_order(evidence)[:max_units]

    @staticmethod
    def _evidence_text(evidence: list[str]) -> str:
        return "\n".join(f"- {item}" for item in evidence)

    def _dag_evidence_token_target_for_budget(self, raw_tokens: int, update_budget_tokens: int) -> int:
        if raw_tokens <= 0:
            return 0
        if raw_tokens <= DAG_EVIDENCE_FULL_KEEP_TOKENS:
            return raw_tokens
        target = int(round(raw_tokens * DAG_EVIDENCE_TARGET_RATIO))
        previous_budget = self._active_compression_update_budget_tokens
        self._active_compression_update_budget_tokens = max(1, int(update_budget_tokens))
        try:
            target = min(self._dynamic_evidence_max_tokens(), target)
        finally:
            self._active_compression_update_budget_tokens = previous_budget
        return min(raw_tokens, max(DAG_EVIDENCE_MIN_TOKENS, target))

    def _dag_evidence_token_target(self, raw_tokens: int) -> int:
        return self._dag_evidence_token_target_for_budget(raw_tokens, self._current_compression_update_budget_tokens())

    def _thin_evidence_with_dag(
        self,
        evidence: list[str],
        block: CompressionBlock,
        semantic_dag: TurnSemanticDAG | None,
    ) -> list[str]:
        if not evidence or semantic_dag is None:
            return evidence
        raw_tokens = estimate_text_tokens(self._evidence_text(evidence))
        target_tokens = self._dag_evidence_token_target(raw_tokens)
        if target_tokens <= 0 or raw_tokens <= target_tokens:
            return evidence

        ranked: list[tuple[float, int, int, str]] = []
        score_by_index: dict[int, float] = {}
        for index, unit in enumerate(evidence):
            cleaned = self._clean_fact(unit)
            if not cleaned:
                continue
            score = self._score_evidence_unit_for_dag(cleaned, unit, block, semantic_dag)
            score_by_index[index] = score
            ranked.append((score, -index, index, unit))
        if not ranked:
            return evidence

        ranked.sort(reverse=True)
        protected_indices: set[int] = {
            index
            for index, unit in enumerate(evidence)
            if block.block_type == "topic" and self._is_dialogue_meta_evidence(unit)
        }
        selected_indices: set[int] = set(protected_indices)
        for _, _, index, unit in ranked:
            trial_indices = sorted({*selected_indices, index})
            trial = [evidence[item_index] for item_index in trial_indices]
            trial_tokens = estimate_text_tokens(self._evidence_text(trial))
            if trial_tokens <= target_tokens or not selected_indices:
                selected_indices.add(index)
                continue
            if estimate_text_tokens(self._evidence_text([evidence[item_index] for item_index in sorted(selected_indices)])) >= target_tokens:
                break

        if not selected_indices:
            best_unit = ranked[0][3]
            return [self._truncate_segment_to_token_cap(best_unit, target_tokens)]

        selected = [evidence[index] for index in sorted(selected_indices)]
        selected_tokens = estimate_text_tokens(self._evidence_text(selected))
        while selected_tokens > max(target_tokens, 1) and len(selected_indices) > 1:
            removable = [index for index in selected_indices if index not in protected_indices]
            if not removable:
                break
            drop_index = min(removable, key=lambda index: (score_by_index.get(index, 0.0), -index))
            selected_indices.remove(drop_index)
            selected = [evidence[index] for index in sorted(selected_indices)]
            selected_tokens = estimate_text_tokens(self._evidence_text(selected))
        if selected_tokens <= max(target_tokens, 1):
            return selected
        if len(selected) == 1:
            return [self._truncate_segment_to_token_cap(selected[0], target_tokens)]
        return selected[:-1] or [self._truncate_segment_to_token_cap(selected[0], target_tokens)]

    def _score_evidence_unit_for_dag(
        self,
        cleaned: str,
        raw_unit: str,
        block: CompressionBlock,
        semantic_dag: TurnSemanticDAG,
    ) -> float:
        node_type = self._classify_semantic_node(cleaned)
        methods = self._extract_methods(cleaned)
        objects = self._extract_objects(cleaned)
        attributes = self._extract_attributes(cleaned)
        keywords = self._extract_keywords(cleaned)
        score = self._semantic_salience(cleaned, node_type, methods, objects, attributes)
        if raw_unit.lstrip().startswith("用户"):
            score += 0.45 if block.block_type == "topic" else 0.18
        if block.block_type == "topic" and self._conversation_memory_query_intent(cleaned):
            score += 0.65
        if "？" in raw_unit or "?" in raw_unit:
            score += 0.08
        if any(self._term_matches(term, cleaned) for term in block.protected_terms):
            score += 0.35
        if any(keyword and self._term_matches(keyword, cleaned) for keyword in block.keywords):
            score += 0.25

        unit_keywords = set(keywords)
        for node in semantic_dag.nodes:
            if cleaned == node.text or cleaned in node.text or node.text in cleaned:
                score = max(score, node.salience + 0.35)
                continue
            shared_keywords = unit_keywords & set(node.keywords)
            if shared_keywords:
                score = max(score, node.salience + min(0.3, len(shared_keywords) * 0.08))
            elif self._shares_meaningful_fragment(cleaned, node.text):
                score = max(score, node.salience + 0.12)
        return score

    @staticmethod
    def _node_line_start(node: SemanticDAGNode) -> int:
        if not node.anchors:
            return 0
        return node.anchors[0].line_start

    @staticmethod
    def _node_line_range(node: SemanticDAGNode) -> list[int]:
        if not node.anchors:
            return [0, 0]
        return [node.anchors[0].line_start, node.anchors[0].line_end]

    def _compression_dag_material_payload(
        self,
        semantic_dag: TurnSemanticDAG,
        block: CompressionBlock,
    ) -> dict:
        del block
        dags_by_turn = {dag.turn_id: dag for dag in self.turn_dag_cache}
        dags_by_turn[semantic_dag.turn_id] = semantic_dag
        direct_turns = [
            turn_id
            for turn_id in sorted(dags_by_turn)
            if semantic_dag.turn_id - turn_id < DAG_MATERIAL_DIRECT_HISTORY_TURNS
        ]
        materials = [
            self._make_dag_material_for_turn(dags_by_turn[turn_id], current_turn_id=semantic_dag.turn_id)
            for turn_id in direct_turns
        ]
        history_turns = [
            dags_by_turn[turn_id]
            for turn_id in sorted(dags_by_turn)
            if turn_id not in set(direct_turns)
        ]
        history_sketch = self._build_dag_history_sketch(history_turns, current_turn_id=semantic_dag.turn_id)
        return {
            "current_turn_id": semantic_dag.turn_id,
            "source": "lossy_dag_material_cache",
            "rule": f"按 exp(-{DAG_MATERIAL_DECAY_RATE}*age) 衰减；直接材料逐轮换出，远端历史进入滚动 sketch。",
            "materials": [material.payload() for material in materials],
            "history_sketch": history_sketch,
        }

    @staticmethod
    def _block_update_payload(block: CompressionBlock, max_tokens: int | None = None) -> dict:
        facts = list(block.facts)
        risks = list(block.risks)

        def make_payload() -> dict:
            return {
                "id": block.id,
                "标题": block.title,
                "要点": facts,
                "风险": risks,
                "stability": block.stability,
                "depends_on": block.depends_on,
                "supports": block.supports,
            }

        payload = make_payload()
        if max_tokens is None or estimate_text_tokens(json.dumps(payload, ensure_ascii=False)) <= max_tokens:
            return payload

        while len(facts) > 1 and estimate_text_tokens(json.dumps(make_payload(), ensure_ascii=False)) > max_tokens:
            archive = [fact for fact in facts if fact.startswith("早期摘要：")]
            regular = [fact for fact in facts if not fact.startswith("早期摘要：")]
            if archive and len(regular) > 1:
                regular = regular[1:]
                facts = [archive[0], *regular]
            else:
                facts = facts[1:]

        while risks and estimate_text_tokens(json.dumps(make_payload(), ensure_ascii=False)) > max_tokens:
            risks = risks[1:]

        payload = make_payload()
        if estimate_text_tokens(json.dumps(payload, ensure_ascii=False)) <= max_tokens:
            return payload
        return {
            "id": block.id,
            "标题": block.title,
            "要点": facts[-1:] if facts else [],
            "风险": risks[-1:] if risks else [],
            "stability": block.stability,
            "depends_on": block.depends_on,
            "supports": block.supports,
        }

    def compress_turn(
        self,
        raw_turn: str,
        turn_id: int,
        llm_generate: LLMGenerator,
        max_dirty_blocks: int = 2,
        target_tokens_per_block: int = 180,
    ) -> CompressionTurnReport:
        semantic_dag = self.index_turn_memory(raw_turn, turn_id)
        report = self.compress_turn_blocks(
            raw_turn,
            turn_id=turn_id,
            llm_generate=llm_generate,
            max_dirty_blocks=max_dirty_blocks,
            target_tokens_per_block=target_tokens_per_block,
            semantic_dag=semantic_dag,
        )
        return report

    def compress_turn_blocks(
        self,
        raw_turn: str,
        turn_id: int,
        llm_generate: LLMGenerator,
        max_dirty_blocks: int = 2,
        target_tokens_per_block: int = 180,
        semantic_dag: TurnSemanticDAG | None = None,
    ) -> CompressionTurnReport:
        routed_blocks = self.route_dirty_blocks(raw_turn, max_blocks=max_dirty_blocks)
        updates = [
            self.update_block(
                block_id=block_id,
                raw_turn=raw_turn,
                turn_id=turn_id,
                llm_generate=llm_generate,
                target_tokens=target_tokens_per_block,
                semantic_dag=semantic_dag,
            )
            for block_id in routed_blocks
        ]
        rendered_context = self.render()
        return CompressionTurnReport(
            turn_id=turn_id,
            routed_blocks=routed_blocks,
            updates=updates,
            rendered_tokens=estimate_text_tokens(rendered_context),
            rendered_context=rendered_context,
        )

    def update_block(
        self,
        block_id: str,
        raw_turn: str,
        turn_id: int,
        llm_generate: LLMGenerator,
        target_tokens: int = 180,
        semantic_dag: TurnSemanticDAG | None = None,
    ) -> CompressionUpdateReport:
        block = self.blocks[block_id]
        old_hash = block.content_hash
        raw_evidence_max_units = DAG_EVIDENCE_SOURCE_MAX_UNITS if semantic_dag is not None else 8
        raw_evidence = self.filter_evidence(raw_turn, block_id, max_units=raw_evidence_max_units)
        raw_evidence_tokens = estimate_text_tokens(self._evidence_text(raw_evidence)) if raw_evidence else 0
        update_budget_tokens = self._compression_update_posterior_budget(
            turn_id,
            raw_evidence_tokens,
            semantic_dag,
            update_state=True,
        )
        evidence = self._thin_evidence_with_dag(raw_evidence, block, semantic_dag)
        evidence_tokens = estimate_text_tokens(self._evidence_text(evidence)) if evidence else 0
        evidence_target_tokens = self._dag_evidence_token_target(raw_evidence_tokens)
        old_block_payload = self._block_update_payload(block, max_tokens=self._dynamic_old_block_max_tokens())
        old_block_tokens = estimate_text_tokens(json.dumps(old_block_payload, ensure_ascii=False))
        dag_material_tokens = 0
        if semantic_dag is not None:
            dag_payload = self._compression_dag_material_payload(semantic_dag, block)
            dag_json = json.dumps(dag_payload, ensure_ascii=False, separators=(",", ":"))
            dag_material_tokens = estimate_text_tokens(dag_json)
        report = CompressionUpdateReport(
            block_id=block_id,
            status="skipped",
            update_budget_tokens=update_budget_tokens,
            evidence_tokens=evidence_tokens,
            raw_evidence_tokens=raw_evidence_tokens,
            evidence_target_tokens=evidence_target_tokens,
            old_block_tokens=old_block_tokens,
            dag_material_tokens=dag_material_tokens,
            old_hash=old_hash,
        )

        if not evidence:
            return report

        cache_key = self._cache_key(block, evidence, target_tokens, semantic_dag=semantic_dag)
        if cache_key in self.block_cache:
            self.blocks[block_id] = copy.deepcopy(self.block_cache[cache_key])
            report.status = "cache_hit"
            report.cache_hit = True
            report.new_hash = self.blocks[block_id].content_hash
            return report

        prompt = self._build_update_prompt(block, evidence, target_tokens, semantic_dag=semantic_dag)
        response = llm_generate(prompt, target_tokens)
        report.model_calls = 1
        report.stats = response.stats

        candidate, errors = self._parse_candidate(response.text, block)
        if candidate is None:
            patched = self._deterministic_patch(block, evidence, turn_id)
            report.fallback_used = True
            report.validation_errors = errors
        else:
            patched, patch_errors = self._validate_and_merge(block, candidate, evidence, turn_id)
            if patch_errors:
                report.fallback_used = True
                report.validation_errors = [*errors, *patch_errors]

        self.blocks[block_id] = patched
        self._mark_dependents_stale(block_id)
        self.block_cache[cache_key] = copy.deepcopy(patched)
        report.status = "updated"
        report.new_hash = patched.content_hash
        return report

    def render(self) -> str:
        sections: list[str] = []
        budget_left = self.token_budget
        for block in self._ordered_blocks_for_render():
            section = self._render_block(block)
            if not section:
                continue
            tokens = estimate_text_tokens(section)
            if tokens <= budget_left:
                sections.append(section)
                budget_left -= tokens
                continue
            trimmed = self._trim_block_to_budget(block, budget_left)
            if trimmed:
                sections.append(trimmed)
            break
        return "\n\n".join(sections).strip()

    def dependency_edges(self) -> list[tuple[str, str]]:
        edges: list[tuple[str, str]] = []
        for block in self.blocks.values():
            for dependency in block.depends_on:
                edges.append((dependency, block.id))
        return edges

    def snapshot(self) -> dict:
        return {
            "policy_version": self.policy_version,
            "token_budget": self.token_budget,
            "compression_update_token_budget": self.compression_update_token_budget,
            "compression_update_posterior_budget": self._current_compression_update_budget_tokens(),
            "blocks": {block_id: block.payload() for block_id, block in self.blocks.items()},
            "dependencies": self.dependency_edges(),
            "raw_memory_turns": len(self.turn_dag_cache),
            "dag_material_turns": len(self.dag_material_cache),
        }

    def raw_memory_snapshot(self) -> list[dict]:
        return [dag.payload() for dag in self.turn_dag_cache]

    def index_turn_memory(self, raw_turn: str, turn_id: int) -> TurnSemanticDAG:
        dag = self._build_turn_dag(raw_turn, turn_id)
        self.turn_dag_cache = [cached for cached in self.turn_dag_cache if cached.turn_id != turn_id]
        self.turn_dag_cache.append(dag)
        self.turn_dag_cache.sort(key=lambda item: item.turn_id)
        self._upsert_dag_material(dag)
        return dag

    def _upsert_dag_material(self, dag: TurnSemanticDAG) -> None:
        self.dag_material_cache = [item for item in self.dag_material_cache if item.turn_id != dag.turn_id]
        self.dag_material_cache.append(self._make_dag_material_for_turn(dag, current_turn_id=dag.turn_id))
        self.dag_material_cache.sort(key=lambda item: item.turn_id)
        self._age_dag_material_cache()

    def _age_dag_material_cache(self) -> None:
        if not self.dag_material_cache:
            return
        current_turn_id = max(item.turn_id for item in self.dag_material_cache)
        dag_by_turn = {dag.turn_id: dag for dag in self.turn_dag_cache}
        aged: list[DAGCompressionMaterial] = []
        for material in self.dag_material_cache:
            dag = dag_by_turn.get(material.turn_id)
            if dag is None:
                aged.append(material)
                continue
            aged.append(self._make_dag_material_for_turn(dag, current_turn_id=current_turn_id))
        self.dag_material_cache = aged

    @staticmethod
    def _dag_material_decay_weight(age: int) -> float:
        return math.exp(-DAG_MATERIAL_DECAY_RATE * max(0, age))

    @staticmethod
    def _dag_material_precision(age: int, decay_weight: float) -> str:
        if age < DAG_MATERIAL_RECENT_FULL_TURNS:
            return "recent"
        if decay_weight >= DAG_MATERIAL_TRACE_WEIGHT:
            return "aged"
        return "trace"

    def _make_dag_material_for_turn(self, dag: TurnSemanticDAG, current_turn_id: int) -> DAGCompressionMaterial:
        age = max(0, current_turn_id - dag.turn_id)
        decay_weight = self._dag_material_decay_weight(age)
        precision = self._dag_material_precision(age, decay_weight)
        return self._make_dag_material(dag, precision=precision, age=age, decay_weight=decay_weight)

    def _dag_material_node_limit(self, precision: str, decay_weight: float) -> int:
        if precision == "recent":
            return self._dynamic_recent_dag_node_limit()
        if precision == "trace":
            return DAG_MATERIAL_TRACE_MAX_NODES
        aged_limit = self._dynamic_aged_dag_node_limit()
        scaled = int(round(aged_limit * max(0.18, decay_weight * 2.0)))
        return max(1, min(aged_limit, scaled))

    def _build_dag_history_sketch(self, dags: list[TurnSemanticDAG], current_turn_id: int) -> dict:
        if not dags:
            return {}
        candidates: list[tuple[float, int, dict[str, Any]]] = []
        for dag in dags:
            age = max(0, current_turn_id - dag.turn_id)
            decay_weight = self._dag_material_decay_weight(age)
            if not dag.nodes:
                continue
            node = max(dag.nodes, key=lambda item: (item.salience, len(item.keywords), -self._node_line_start(item)))
            weighted_salience = round(node.salience * decay_weight, 4)
            candidates.append(
                (
                    weighted_salience,
                    dag.turn_id,
                    {
                        "turn_id": dag.turn_id,
                        "age": age,
                        "decay": round(decay_weight, 4),
                        "t": node.node_type,
                        "s": self._shorten_dag_material_text(
                            node.summary or node.text,
                            max_chars=DAG_MATERIAL_HISTORY_SKETCH_SUMMARY_MAX_CHARS,
                        ),
                        "k": node.keywords[:1],
                        "w": weighted_salience,
                    },
                )
            )
        selected = [
            item
            for _, _, item in sorted(candidates, key=lambda entry: (entry[0], entry[1]), reverse=True)[
                : self._dynamic_history_sketch_item_limit()
            ]
        ]
        selected.sort(key=lambda item: item["turn_id"])
        turn_ids = [dag.turn_id for dag in dags]
        return {
            "turn_span": [min(turn_ids), max(turn_ids)],
            "turn_count": len(turn_ids),
            "items": selected,
        }

    def _make_dag_material(
        self,
        dag: TurnSemanticDAG,
        precision: str,
        age: int = 0,
        decay_weight: float = 1.0,
    ) -> DAGCompressionMaterial:
        max_nodes = self._dag_material_node_limit(precision, decay_weight)
        nodes = sorted(
            dag.nodes,
            key=lambda node: (node.salience, len(node.keywords), -self._node_line_start(node)),
            reverse=True,
        )[:max_nodes]
        nodes.sort(key=self._node_line_start)
        local_ids = {node.node_id: f"n{index + 1}" for index, node in enumerate(nodes)}
        include_details = precision == "recent"
        summary_chars = {
            "recent": DAG_MATERIAL_SUMMARY_MAX_CHARS,
            "aged": DAG_MATERIAL_AGED_SUMMARY_MAX_CHARS,
            "trace": DAG_MATERIAL_TRACE_SUMMARY_MAX_CHARS,
        }.get(precision, DAG_MATERIAL_TRACE_SUMMARY_MAX_CHARS)
        material_nodes: list[dict[str, Any]] = []
        for node in nodes:
            weighted_salience = round(node.salience * decay_weight, 3)
            if precision == "trace":
                material_nodes.append(
                    {
                        "t": node.node_type,
                        "s": self._shorten_dag_material_text(node.summary or node.text, max_chars=summary_chars),
                        "k": node.keywords[:1],
                        "w": weighted_salience,
                    }
                )
                continue
            material_nodes.append(
                {
                    "i": local_ids[node.node_id],
                    "t": node.node_type,
                    "s": self._shorten_dag_material_text(node.summary or node.text, max_chars=summary_chars),
                    "k": node.keywords[:4 if include_details else 2],
                    "m": node.methods[:2] if include_details else [],
                    "o": node.objects[:2] if include_details else [],
                    "a": node.attributes[:2] if include_details else [],
                    "loc": self._node_line_range(node),
                    "w": weighted_salience,
                }
            )
        material_edges = []
        if precision == "recent":
            material_edges = [
                [local_ids[edge.source], local_ids[edge.target], edge.relation, round(edge.weight * decay_weight, 3)]
                for edge in sorted(dag.edges, key=lambda item: item.weight, reverse=True)
                if edge.source in local_ids and edge.target in local_ids
            ][: self._dynamic_recent_dag_edge_limit()]
        return DAGCompressionMaterial(
            turn_id=dag.turn_id,
            precision=precision,
            age=age,
            decay_weight=decay_weight,
            nodes=material_nodes,
            edges=material_edges,
            created_at=dag.created_at,
        )

    @staticmethod
    def _shorten_dag_material_text(text: str, max_chars: int = DAG_MATERIAL_SUMMARY_MAX_CHARS) -> str:
        normalized = _normalize_space(text)
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 1].rstrip("，,；;。 ") + "…"

    def backtracking_decision(self, query: str) -> BacktrackingDecision:
        normalized = _normalize_space(query)
        if not normalized:
            return BacktrackingDecision(triggered=False)

        reasons: list[str] = []
        lowered = normalized.lower()
        for marker in self.locale.list("referential", "markers"):
            if marker and (marker in normalized or marker.lower() in lowered):
                reasons.append(f"referential_marker:{marker}")
                break

        if any(marker and (marker in normalized or marker.lower() in lowered) for marker in self.locale.list("referential", "time_reference_markers")):
            reasons.append("time_reference")

        for pattern in self.locale.list("referential", "missing_subject_patterns"):
            try:
                if re.search(pattern, normalized, flags=re.I):
                    reasons.append("missing_subject_pattern")
                    break
            except re.error:
                continue

        follow_up_prefixes = tuple(self.locale.list("referential", "short_follow_up_prefixes"))
        max_chars = int(self.locale.section("referential").get("short_follow_up_max_chars", 18) or 18)
        bare_follow_up = bool(follow_up_prefixes) and normalized.lower().startswith(
            tuple(prefix.lower() for prefix in follow_up_prefixes)
        ) and len(normalized) <= max_chars
        if bare_follow_up:
            reasons.append("short_follow_up")

        return BacktrackingDecision(triggered=bool(reasons), reasons=self._dedupe_keep_order(reasons))

    def retrieve_backtracking_context(
        self,
        query: str,
        recent_turns: int = BACKTRACK_RECENT_DAG_TURNS,
        top_k: int = BACKTRACK_TOP_K_NODES,
        max_hits: int = BACKTRACK_MAX_HITS,
        expand_lines: int = BACKTRACK_EXPAND_LINES,
    ) -> BacktrackingRetrievalResult:
        decision = self.backtracking_decision(query)
        if not decision.triggered or not self.turn_dag_cache:
            return BacktrackingRetrievalResult(decision=decision)

        query_keywords = self._extract_keywords(query)
        preferred_types = self._preferred_node_types(query)
        candidate_turns = self.turn_dag_cache[-max(1, recent_turns) :]
        candidates: list[RetrievalCandidate] = []

        for recency_index, dag in enumerate(reversed(candidate_turns), start=1):
            recency_weight = max(0.2, 1.0 - (recency_index - 1) * 0.14)
            node_map = {node.node_id: node for node in dag.nodes}
            for node in dag.nodes:
                matched_keywords = self._matched_keywords(query_keywords, node)
                overlap_score = float(len(matched_keywords))
                fragment_bonus = 1.2 if self._shares_meaningful_fragment(query, node.text) else 0.0
                type_bonus = 1.3 if node.node_type in preferred_types else 0.0
                fallback_bonus = 0.7 if not matched_keywords and decision.triggered else 0.0
                score = overlap_score * 1.8 + fragment_bonus + type_bonus + fallback_bonus + recency_weight + node.salience * 0.25
                if score <= 1.15:
                    continue
                confidence = min(
                    0.99,
                    0.25
                    + min(0.35, overlap_score * 0.12)
                    + min(0.2, type_bonus * 0.1)
                    + min(0.2, recency_weight * 0.18),
                )
                neighbors = self._neighbor_node_ids(dag, node.node_id, node_map=node_map)
                candidates.append(
                    RetrievalCandidate(
                        turn_id=dag.turn_id,
                        node_id=node.node_id,
                        node_type=node.node_type,
                        text=node.text,
                        score=score,
                        confidence=confidence,
                        matched_keywords=matched_keywords,
                        anchors=list(node.anchors),
                        neighbor_node_ids=neighbors,
                    )
                )

        candidates.sort(key=lambda item: (item.score, item.confidence, item.turn_id), reverse=True)
        candidates = self._dedupe_candidates(candidates)[: max(1, top_k)]
        segments = self._saturate_candidates(query_keywords, candidates, max_hits=max_hits, expand_lines=expand_lines)
        fused_context = self._render_backtracking_context(decision, candidates, segments)
        return BacktrackingRetrievalResult(
            decision=decision,
            candidates=candidates,
            segments=segments,
            fused_context=fused_context,
        )

    def build_backtracking_prompt_context(self, query: str) -> str:
        return self.retrieve_backtracking_context(query).fused_context

    def _build_turn_dag(self, raw_turn: str, turn_id: int) -> TurnSemanticDAG:
        lines = raw_turn.splitlines() or [raw_turn]
        chunks = self._chunk_raw_lines(lines, turn_id)
        nodes: list[SemanticDAGNode] = []
        edges: list[SemanticDAGEdge] = []
        compressed_nodes: list[CompressedDAGNode] = []

        previous_node: SemanticDAGNode | None = None
        for chunk in chunks:
            chunk_lines = lines[chunk.line_start - 1 : chunk.line_end]
            for offset, line in enumerate(chunk_lines, start=chunk.line_start):
                cleaned = self._clean_fact(line)
                if not cleaned:
                    continue
                node_type = self._classify_semantic_node(cleaned)
                anchor = SourceAnchor(
                    turn_id=turn_id,
                    chunk_id=chunk.chunk_id,
                    line_start=offset,
                    line_end=offset,
                )
                methods = self._extract_methods(cleaned)
                objects = self._extract_objects(cleaned)
                attributes = self._extract_attributes(cleaned)
                keywords = self._extract_keywords(cleaned)
                node_id = self._stable_node_id(turn_id, chunk.chunk_id, offset, cleaned)
                summary = self._compress_semantic_summary(cleaned, keywords, node_type)
                salience = self._semantic_salience(cleaned, node_type, methods, objects, attributes)
                node = SemanticDAGNode(
                    node_id=node_id,
                    node_type=node_type,
                    text=cleaned,
                    summary=summary,
                    keywords=keywords,
                    methods=methods,
                    objects=objects,
                    attributes=attributes,
                    anchors=[anchor],
                    confidence=1.0,
                    salience=salience,
                )
                nodes.append(node)
                compressed_nodes.append(
                    CompressedDAGNode(
                        node_id=node_id,
                        node_type=node_type,
                        summary=summary,
                        anchor_refs=[anchor],
                    )
                )
                if previous_node is not None:
                    relation, weight = self._infer_edge_relation(previous_node, node)
                    edges.append(
                        SemanticDAGEdge(
                            source=previous_node.node_id,
                            target=node.node_id,
                            relation=relation,
                            weight=weight,
                        )
                    )
                previous_node = node

        node_map = {node.node_id: node for node in nodes}
        for left_index, left in enumerate(nodes):
            for right in nodes[left_index + 1 : left_index + 4]:
                relation, weight = self._infer_nonlocal_relation(left, right)
                if relation:
                    edges.append(SemanticDAGEdge(source=left.node_id, target=right.node_id, relation=relation, weight=weight))

        anchors_by_node = {node_id: list(node.anchors) for node_id, node in node_map.items()}
        compressed_nodes = self._merge_compressed_nodes(compressed_nodes, anchors_by_node)
        return TurnSemanticDAG(
            turn_id=turn_id,
            raw_text=raw_turn,
            lines=lines,
            chunks=chunks,
            nodes=nodes,
            edges=edges,
            compressed_nodes=compressed_nodes,
        )

    def _chunk_raw_lines(self, lines: list[str], turn_id: int) -> list[RawTextChunk]:
        chunks: list[RawTextChunk] = []
        chunk_lines: list[str] = []
        chunk_start = 1
        for line_index, raw_line in enumerate(lines, start=1):
            normalized = raw_line.rstrip()
            if not chunk_lines:
                chunk_start = line_index
            chunk_lines.append(normalized)
            joined = "\n".join(chunk_lines).strip()
            if (
                len(chunk_lines) >= RAW_MEMORY_CHUNK_MAX_LINES
                or len(joined) >= RAW_MEMORY_CHUNK_MAX_CHARS
                or line_index == len(lines)
            ):
                chunk_text = joined
                digest = hashlib.sha1(
                    f"{turn_id}:{chunk_start}:{line_index}:{chunk_text}".encode("utf-8")
                ).hexdigest()[:10]
                chunks.append(
                    RawTextChunk(
                        chunk_id=f"t{turn_id:04d}-c{len(chunks) + 1:02d}-{digest}",
                        turn_id=turn_id,
                        ordinal=len(chunks) + 1,
                        line_start=chunk_start,
                        line_end=line_index,
                        text=chunk_text,
                    )
                )
                chunk_lines = []
        return chunks

    @staticmethod
    def _stable_node_id(turn_id: int, chunk_id: str, line_no: int, text: str) -> str:
        digest = hashlib.sha1(f"{turn_id}:{chunk_id}:{line_no}:{text}".encode("utf-8")).hexdigest()[:10]
        return f"t{turn_id:04d}-n{line_no:02d}-{digest}"

    def _classify_semantic_node(self, text: str) -> str:
        lowered = text.lower()
        preference_terms = [
            *self.locale.list("preference", "concrete_terms"),
            *self.locale.list("preference", "preference_terms"),
            "喜欢",
            "不喜欢",
            "想吃",
            "不吃",
            "清淡",
            "微辣",
            "prefer",
            "preference",
            "diet",
            "taste",
        ]
        if any(self._term_matches(token, text) for token in preference_terms):
            return "preference"
        if any(self._term_matches(token, text) for token in ["风险", "泄露", "攻击", "注入", "失败", "兜底", "risk", "leak", "attack", "inject", "fail", "fallback"]):
            return "risk"
        if any(self._term_matches(token, text) for token in ["配置", "参数", "上限", "tokens", "setting", "configuration", "step_size", "K="]):
            return "config"
        if any(self._term_matches(token, text) for token in ["作废", "采用", "确认", "正式", "决策", "必须", "decide", "decision", "confirm", "adopt", "must"]):
            return "decision"
        if any(self._term_matches(token, lowered) for token in ["how", "why", "because", "method", "implement"]) or any(
            self._term_matches(token, text) for token in ["如何", "怎么", "实现", "通过", "采用", "机制", "步骤", "方法"]
        ):
            return "method"
        if any(self._term_matches(token, text) for token in ["目标", "主题", "对象", "问题", "讨论", "结论", "goal", "topic", "object", "problem", "discuss", "conclusion"]):
            return "object"
        return "fact"

    def _extract_methods(self, text: str) -> list[str]:
        methods: list[str] = []
        for marker in ["通过", "采用", "使用", "实现", "方法", "机制", "步骤"]:
            if marker in text:
                suffix = text.split(marker, 1)[1].strip("：: ，。；;")
                if suffix:
                    methods.append(f"{marker}{suffix[:24]}")
        methods.extend(re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b", text))
        return self._dedupe_keep_order(methods)[:6]

    def _extract_objects(self, text: str) -> list[str]:
        objects: list[str] = []
        for marker in ["关于", "围绕", "讨论", "主题", "对象", "目标", "问题", "系统"]:
            if marker in text:
                suffix = text.split(marker, 1)[1].strip("：: ，。；;")
                if suffix:
                    objects.append(suffix[:24])
        return self._dedupe_keep_order(objects or self._extract_keywords(text)[:4])[:6]

    def _extract_attributes(self, text: str) -> list[str]:
        attributes: list[str] = []
        for marker in ["必须", "需要", "不能", "上限", "最多", "保留", "删除", "合并", "扩展"]:
            if marker in text:
                attributes.append(marker)
        if re.search(r"\b\d+\s*(?:token|tokens|GB|行)\b", text, flags=re.I):
            attributes.append("numeric_bound")
        return self._dedupe_keep_order(attributes)[:6]

    def _compress_semantic_summary(self, text: str, keywords: list[str], node_type: str) -> str:
        summary = self._clean_fact(text)
        if len(summary) <= 60:
            return summary
        if keywords:
            head = " / ".join(keywords[:3])
            return f"{node_type}:{head}"
        return summary[:57].rstrip() + "..."

    @staticmethod
    def _semantic_salience(
        text: str,
        node_type: str,
        methods: list[str],
        objects: list[str],
        attributes: list[str],
    ) -> float:
        base = {
            "method": 1.0,
            "decision": 0.95,
            "risk": 0.92,
            "config": 0.88,
            "object": 0.8,
            "preference": 0.72,
            "fact": 0.68,
        }.get(node_type, 0.6)
        base += min(0.15, len(methods) * 0.03)
        base += min(0.12, len(objects) * 0.02)
        base += min(0.1, len(attributes) * 0.02)
        if len(text) > 45:
            base += 0.03
        return min(1.3, base)

    @staticmethod
    def _infer_edge_relation(previous_node: SemanticDAGNode, node: SemanticDAGNode) -> tuple[str, float]:
        if previous_node.node_type == node.node_type:
            return "sequence_same_type", 0.8
        if previous_node.node_type == "object" and node.node_type == "method":
            return "implements", 0.92
        if node.node_type == "risk":
            return "risk_of", 0.88
        if node.node_type == "decision":
            return "concludes", 0.86
        return "sequence", 0.7

    def _infer_nonlocal_relation(self, left: SemanticDAGNode, right: SemanticDAGNode) -> tuple[str | None, float]:
        shared = set(left.keywords) & set(right.keywords)
        if shared:
            relation = "same_subject" if left.node_type == right.node_type else "references"
            return relation, min(0.95, 0.6 + len(shared) * 0.08)
        if left.node_type == "method" and right.node_type == "decision":
            return "supports", 0.72
        return None, 0.0

    def _merge_compressed_nodes(
        self,
        nodes: list[CompressedDAGNode],
        anchors_by_node: dict[str, list[SourceAnchor]],
    ) -> list[CompressedDAGNode]:
        merged_by_key: dict[tuple[str, str], CompressedDAGNode] = {}
        for node in nodes:
            key = (node.node_type, node.summary)
            if key not in merged_by_key:
                merged_by_key[key] = copy.deepcopy(node)
                continue
            existing = merged_by_key[key]
            existing.anchor_refs = self._merge_anchor_lists(existing.anchor_refs, node.anchor_refs)
        for node in merged_by_key.values():
            if node.node_id in anchors_by_node:
                node.anchor_refs = self._merge_anchor_lists(node.anchor_refs, anchors_by_node[node.node_id])
        return list(merged_by_key.values())

    @staticmethod
    def _merge_anchor_lists(left: list[SourceAnchor], right: list[SourceAnchor]) -> list[SourceAnchor]:
        merged: list[SourceAnchor] = []
        seen: set[tuple[int, str, int, int]] = set()
        for anchor in [*left, *right]:
            key = (anchor.turn_id, anchor.chunk_id, anchor.line_start, anchor.line_end)
            if key in seen:
                continue
            seen.add(key)
            merged.append(anchor)
        return merged

    def _matched_keywords(self, query_keywords: list[str], node: SemanticDAGNode) -> list[str]:
        if not query_keywords:
            return []
        searchable = "\n".join([node.text, node.summary, *node.keywords, *node.methods, *node.objects, *node.attributes])
        matched = [keyword for keyword in query_keywords if self._term_matches(keyword, searchable)]
        return self._dedupe_keep_order(matched)

    def _preferred_node_types(self, query: str) -> set[str]:
        text = _normalize_space(query)
        preferred: set[str] = set()
        if any(self._term_matches(token, text) for token in ["怎么", "如何", "实现", "方法", "机制", "步骤", "how", "implement", "implementation", "method", "mechanism", "step", "approach"]):
            preferred.update({"method", "decision"})
        if any(self._term_matches(token, text) for token in ["为什么", "为何", "原因", "why", "reason", "because"]):
            preferred.update({"decision", "risk", "object"})
        if any(self._term_matches(token, text) for token in ["配置", "参数", "上限", "token", "K", "setting", "configuration", "limit", "parameter"]):
            preferred.add("config")
        if any(self._term_matches(token, text) for token in ["风险", "攻击", "泄露", "兜底", "risk", "attack", "leak", "fallback"]):
            preferred.add("risk")
        if any(self._term_matches(token, text) for token in ["偏好", "喜欢", "不吃", "口味", "preference", "prefer", "diet", "taste"]):
            preferred.add("preference")
        if not preferred:
            preferred.update({"method", "object", "decision"})
        return preferred

    def _neighbor_node_ids(
        self,
        dag: TurnSemanticDAG,
        node_id: str,
        node_map: dict[str, SemanticDAGNode] | None = None,
    ) -> list[str]:
        del node_map
        neighbors: list[str] = []
        for edge in dag.edges:
            if edge.source == node_id:
                neighbors.append(edge.target)
            elif edge.target == node_id:
                neighbors.append(edge.source)
        return self._dedupe_keep_order(neighbors)[:3]

    @staticmethod
    def _dedupe_candidates(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        kept: list[RetrievalCandidate] = []
        seen: set[tuple[int, str]] = set()
        for candidate in candidates:
            key = (candidate.turn_id, candidate.node_id)
            if key in seen:
                continue
            seen.add(key)
            kept.append(candidate)
        return kept

    def _saturation_nodes_for_candidate(
        self,
        candidate: RetrievalCandidate,
        node_map: dict[str, SemanticDAGNode],
        query_keywords: list[str],
    ) -> list[tuple[SemanticDAGNode, float]]:
        seed = node_map.get(candidate.node_id)
        if seed is None:
            return []

        seed_line = self._node_line_start(seed)
        neighbor_scores: list[tuple[float, SemanticDAGNode]] = []
        for neighbor_id in candidate.neighbor_node_ids:
            neighbor = node_map.get(neighbor_id)
            if neighbor is None:
                continue
            line_distance = abs(self._node_line_start(neighbor) - seed_line)
            coverage = len(self._matched_keywords(query_keywords, neighbor))
            locality = 1.0 if line_distance <= BACKTRACK_EXPAND_LINES + 1 else 0.35
            type_bonus = 0.5 if neighbor.node_type == seed.node_type else 0.0
            score = coverage * 1.2 + locality + type_bonus + neighbor.salience * 0.2
            neighbor_scores.append((score, neighbor))

        neighbor_scores.sort(key=lambda item: (item[0], self._node_line_start(item[1])), reverse=True)
        selected: list[tuple[SemanticDAGNode, float]] = [(seed, candidate.score)]
        for score, neighbor in neighbor_scores[:BACKTRACK_MAX_NEIGHBOR_NODES]:
            selected.append((neighbor, max(candidate.score * 0.72, score)))
        return selected

    def _saturate_candidates(
        self,
        query_keywords: list[str],
        candidates: list[RetrievalCandidate],
        max_hits: int,
        expand_lines: int,
    ) -> list[SaturatedSegment]:
        spans: list[tuple[TurnSemanticDAG, int, int, str, float, list[str]]] = []
        dag_by_turn = {dag.turn_id: dag for dag in self.turn_dag_cache}
        for candidate in candidates:
            dag = dag_by_turn.get(candidate.turn_id)
            if dag is None:
                continue
            node_map = {node.node_id: node for node in dag.nodes}
            for node, score in self._saturation_nodes_for_candidate(candidate, node_map, query_keywords):
                for anchor in node.anchors[:1]:
                    start = max(1, anchor.line_start - expand_lines)
                    end = min(len(dag.lines), anchor.line_end + expand_lines)
                    spans.append((dag, start, end, anchor.chunk_id, score, [node.node_id]))

        merged: list[SaturatedSegment] = []
        spans.sort(key=lambda item: (item[0].turn_id, item[1], item[2]))
        for dag, start, end, chunk_id, score, node_ids in spans:
            if merged and merged[-1].turn_id == dag.turn_id and start <= merged[-1].line_end + 1:
                merged[-1].line_end = max(merged[-1].line_end, end)
                merged[-1].chunk_ids = self._dedupe_keep_order([*merged[-1].chunk_ids, chunk_id])
                merged[-1].source_node_ids = self._dedupe_keep_order([*merged[-1].source_node_ids, *node_ids])
                merged[-1].confidence = max(merged[-1].confidence, score)
                continue
            merged.append(
                SaturatedSegment(
                    turn_id=dag.turn_id,
                    chunk_ids=[chunk_id],
                    line_start=start,
                    line_end=end,
                    text="",
                    confidence=score,
                    source_node_ids=node_ids,
                )
            )

        scored_segments: list[SaturatedSegment] = []
        for segment in merged:
            dag = dag_by_turn[segment.turn_id]
            text = "\n".join(dag.lines[segment.line_start - 1 : segment.line_end]).strip()
            text = self._truncate_segment_to_token_cap(text, BACKTRACK_MAX_SEGMENT_TOKENS)
            coverage = self._segment_coverage(query_keywords, text)
            recency_rank = next(
                (
                    index
                    for index, item in enumerate(reversed(self.turn_dag_cache[-BACKTRACK_RECENT_DAG_TURNS:]), start=1)
                    if item.turn_id == segment.turn_id
                ),
                BACKTRACK_RECENT_DAG_TURNS,
            )
            time_weight = max(0.25, 1.0 - (recency_rank - 1) * 0.16)
            confidence = min(0.99, segment.confidence * 0.55 + time_weight * 0.25 + coverage * 0.2)
            scored_segments.append(
                SaturatedSegment(
                    turn_id=segment.turn_id,
                    chunk_ids=segment.chunk_ids,
                    line_start=segment.line_start,
                    line_end=segment.line_end,
                    text=text,
                    confidence=confidence,
                    source_node_ids=segment.source_node_ids,
                )
            )
        scored_segments.sort(key=lambda item: (item.confidence, item.turn_id, item.line_start), reverse=True)
        deduped: list[SaturatedSegment] = []
        seen_texts: set[str] = set()
        for segment in scored_segments:
            key = _normalize_space(segment.text)
            if not key or key in seen_texts:
                continue
            seen_texts.add(key)
            deduped.append(segment)
            if len(deduped) >= max_hits:
                break
        return deduped[:max_hits]

    def _segment_coverage(self, query_keywords: list[str], text: str) -> float:
        if not query_keywords:
            return 0.4
        matched = sum(1 for keyword in query_keywords if self._term_matches(keyword, text))
        return min(1.0, matched / max(1, len(query_keywords)))

    @staticmethod
    def _truncate_segment_to_token_cap(text: str, token_cap: int) -> str:
        cleaned = _normalize_space(text.replace("\n", "\n").strip())
        if estimate_text_tokens(cleaned) <= token_cap:
            return cleaned
        max_chars = max(40, token_cap * 4)
        return cleaned[: max_chars - 1].rstrip() + "…"

    def _render_backtracking_context(
        self,
        decision: BacktrackingDecision,
        candidates: list[RetrievalCandidate],
        segments: list[SaturatedSegment],
    ) -> str:
        if not decision.triggered or not segments:
            return ""
        lines = ["[回溯检索]"]
        lines.append(f"- 触发原因：{'、'.join(decision.reasons)}")
        if candidates:
            hit_summaries = []
            for candidate in candidates[:BACKTRACK_MAX_HITS]:
                anchor = candidate.anchors[0] if candidate.anchors else None
                location = f"turn={candidate.turn_id}"
                if anchor is not None:
                    location += f" lines={anchor.line_start}-{anchor.line_end}"
                hit_summaries.append(
                    f"{location} type={candidate.node_type} confidence={candidate.confidence:.2f}"
                )
            lines.append(f"- 命中摘要：{'; '.join(hit_summaries)}")
        lines.append("- 高相关原文：")
        for segment in segments:
            segment_lines = [
                f"[片段 turn={segment.turn_id} lines={segment.line_start}-{segment.line_end} "
                f"confidence={segment.confidence:.2f}]",
                segment.text,
            ]
            trial = "\n".join([*lines, *segment_lines])
            if estimate_text_tokens(trial) > BACKTRACK_CONTEXT_TOKEN_BUDGET:
                break
            lines.extend(segment_lines)
        return "\n".join(lines).strip()

    def _build_update_prompt(
        self,
        block: CompressionBlock,
        evidence: list[str],
        target_tokens: int,
        semantic_dag: TurnSemanticDAG | None = None,
    ) -> str:
        old_block = self._block_update_payload(block, max_tokens=self._dynamic_old_block_max_tokens())
        evidence_text = self._evidence_text(evidence)
        dag_section = ""
        if semantic_dag is not None:
            dag_payload = self._compression_dag_material_payload(semantic_dag, block)
            dag_json = json.dumps(dag_payload, ensure_ascii=False, separators=(",", ":"))
            dag_section = (
                "DAG压缩素材(JSON，有损投影)：\n"
                f"{dag_json}\n"
                "DAG素材使用规则：这是来自独立 DAG 线路的有损素材，只用于帮助识别完整语义链、关键对象、方法、属性和边界。"
                "最终压缩上下文只能写自然语言要点，禁止包含结构 ID、行号、JSON 或 DAG 结构字段；"
                "所有要点仍必须能被下方原文证据支持。\n"
            )
        protected = "、".join(block.protected_terms) if block.protected_terms else "无"
        forbidden = "、".join(block.forbidden_terms) if block.forbidden_terms else "无"
        topic_instruction = ""
        if block.block_type == "topic":
            topic_instruction = (
                "开放主题摘要规则：不依赖关键词表，直接根据证据提取本轮真实讨论主题、"
                "用户问题和助手给出的可延续结论；保留专名、公式、方法名和关键边界。"
                "必须先判断用户这一轮到底在问什么，不要只把助手回答里的知识点改写成摘要；"
                "第一条要点优先写成“本轮对话：用户……”式元摘要，保留用户原问题或请求；"
                "如果用户在追问刚刚、前面或先前聊天记录，摘要必须显式保留这个追溯意图和被回顾的内容。"
                "不要把回复格式要求、寒暄或无关自我介绍写入摘要。\n"
            )
        return (
            "你是本地对话上下文的增量压缩器。只输出合法 JSON，不要输出解释。\n"
            "任务：只更新指定 block，禁止扩写到其它主题。\n"
            f"目标 block: {block.id} / {block.title}\n"
            f"{topic_instruction}"
            f"目标输出上限：约 {target_tokens} tokens。\n"
            f"稳定性：{block.stability}。locked block 必须保留既有核心事实，只允许用证据增补，不能重写主线。\n"
            f"必须保护的术语：{protected}\n"
            f"禁止无证据引入的术语：{forbidden}\n"
            "输出 schema：\n"
            '{"id":"<block id>","标题":"<标题>","要点":["短句"],"风险":["短句"]}\n'
            "旧 block：\n"
            f"{json.dumps(old_block, ensure_ascii=False)}\n"
            f"{dag_section}"
            "仅可使用以下证据：\n"
            f"{evidence_text}\n"
        )

    def _parse_candidate(self, text: str, block: CompressionBlock) -> tuple[dict | None, list[str]]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return None, [f"invalid_json:{exc.msg}"]
        if not isinstance(parsed, dict):
            return None, ["json_not_object"]
        errors: list[str] = []
        if parsed.get("id") not in {None, block.id}:
            errors.append("id_mismatch")
        if parsed.get("标题") != block.title:
            errors.append("title_mismatch")
        facts = parsed.get("要点", [])
        risks = parsed.get("风险", [])
        if not isinstance(facts, list) or not all(isinstance(item, str) for item in facts):
            errors.append("facts_not_string_list")
        if not isinstance(risks, list) or not all(isinstance(item, str) for item in risks):
            errors.append("risks_not_string_list")
        if errors:
            return None, errors
        return parsed, []

    def _validate_and_merge(
        self,
        block: CompressionBlock,
        candidate: dict,
        evidence: list[str],
        turn_id: int,
    ) -> tuple[CompressionBlock, list[str]]:
        errors: list[str] = []
        candidate_facts = [self._clean_fact(item) for item in candidate.get("要点", [])]
        candidate_risks = [self._clean_fact(item) for item in candidate.get("风险", [])]
        if block.block_type == "recent":
            candidate_risks = []
        candidate_facts = [item for item in candidate_facts if item and not self._is_low_value_fact(block, item)]
        candidate_risks = [item for item in candidate_risks if item and not self._is_low_value_fact(block, item)]

        unsupported_facts = [
            item
            for item in candidate_facts
            if not self._is_supported_fact(block, item, evidence) and item not in block.facts
        ]
        unsupported_risks = [
            item
            for item in candidate_risks
            if not self._is_supported_fact(block, item, evidence) and item not in block.risks
        ]
        if unsupported_facts or unsupported_risks:
            errors.append("unsupported_candidate_terms")

        if block.stability == "locked" and not self._candidate_preserves_locked_terms(block, candidate_facts, candidate_risks):
            errors.append("locked_terms_not_preserved")

        missing_terms = self._missing_evidence_terms(block, candidate_facts, candidate_risks, evidence)
        if missing_terms:
            errors.append("evidence_terms_omitted:" + ",".join(missing_terms))

        if block.stability == "locked" and not self._locked_update_allowed(block, evidence):
            candidate_facts = []
            candidate_risks = []

        if errors:
            return self._deterministic_patch(block, evidence, turn_id), errors

        merged = copy.deepcopy(block)
        prefer_existing = block.stability != "volatile" or block.id == "user_preference"
        if block.stability == "locked":
            merged.facts = self._merge_with_limit(block.facts, candidate_facts, block.max_facts, prefer_existing=True)
            merged.risks = self._merge_with_limit(block.risks, candidate_risks, block.max_risks, prefer_existing=True)
        elif block.block_type == "topic":
            merged.facts = self._merge_topic_facts(block, candidate_facts, turn_id)
            merged.risks = self._merge_with_limit(block.risks, candidate_risks[:1], block.max_risks, prefer_existing=False)
        else:
            merged.facts = self._merge_with_limit(block.facts, candidate_facts, block.max_facts, prefer_existing=prefer_existing)
            merged.risks = self._merge_with_limit(block.risks, candidate_risks, block.max_risks, prefer_existing=prefer_existing)
        merged.source_turns = self._merge_turns(block.source_turns, turn_id)
        merged.updated_turn = turn_id
        merged.dirty = False
        merged.stale = False
        return merged, []

    def _deterministic_patch(self, block: CompressionBlock, evidence: list[str], turn_id: int) -> CompressionBlock:
        patched = copy.deepcopy(block)
        extracted_facts: list[str] = []
        extracted_risks: list[str] = []
        locked_update_allowed = block.stability != "locked" or self._locked_update_allowed(block, evidence)
        for item in evidence:
            fact = self._clean_fact(item)
            if not fact:
                continue
            if self._is_low_value_fact(block, fact):
                continue
            if not locked_update_allowed:
                continue
            if self._is_unconfirmed_locked_update(block, fact):
                continue
            if not self._is_supported_fact(block, fact, evidence):
                continue
            if block.block_type != "recent" and (
                block.block_type == "risk" or any(keyword in fact for keyword in ["风险", "注入", "泄露", "失败", "兜底"])
            ):
                extracted_risks.append(fact)
            else:
                extracted_facts.append(fact)

        prefer_existing = block.stability != "volatile" or block.id == "user_preference"
        if block.stability == "locked":
            patched.facts = self._merge_with_limit(block.facts, extracted_facts, block.max_facts, prefer_existing=True)
            patched.risks = self._merge_with_limit(block.risks, extracted_risks, block.max_risks, prefer_existing=True)
        elif block.block_type == "topic":
            patched.facts = self._merge_topic_facts(block, extracted_facts, turn_id)
            patched.risks = self._merge_with_limit(block.risks, extracted_risks[:1], block.max_risks, prefer_existing=False)
        else:
            patched.facts = self._merge_with_limit(block.facts, extracted_facts, block.max_facts, prefer_existing=prefer_existing)
            patched.risks = self._merge_with_limit(block.risks, extracted_risks, block.max_risks, prefer_existing=prefer_existing)
        patched.source_turns = self._merge_turns(block.source_turns, turn_id)
        patched.updated_turn = turn_id
        patched.dirty = False
        patched.stale = False
        return patched

    def _is_supported_fact(self, block: CompressionBlock, fact: str, evidence: list[str]) -> bool:
        evidence_text = "\n".join(evidence)
        for term in block.forbidden_terms:
            if self._term_matches(term, fact):
                return False
        fact_terms = [term for term in [*block.keywords, *block.protected_terms] if term]
        if any(self._term_matches(term, fact) and self._term_matches(term, evidence_text) for term in fact_terms):
            return True
        if any(self._shares_meaningful_fragment(fact, item) for item in evidence):
            return True
        return False

    def _candidate_preserves_locked_terms(
        self,
        block: CompressionBlock,
        candidate_facts: list[str],
        candidate_risks: list[str],
    ) -> bool:
        combined = "\n".join([*block.facts, *block.risks, *candidate_facts, *candidate_risks])
        return all(term in combined for term in block.protected_terms)

    def _missing_evidence_terms(
        self,
        block: CompressionBlock,
        candidate_facts: list[str],
        candidate_risks: list[str],
        evidence: list[str],
    ) -> list[str]:
        evidence_text = "\n".join(evidence)
        merged_text = "\n".join([*block.facts, *block.risks, *candidate_facts, *candidate_risks])
        if block.block_type == "recent":
            identifiers = re.findall(r"\b(?:checkpoint-\d+|fact\d+)\b", evidence_text, flags=re.I)
            return [
                term
                for term in self._dedupe_keep_order(identifiers)
                if term not in merged_text
            ]
        if block.block_type == "topic":
            for item in evidence:
                if not self._is_dialogue_meta_evidence(item):
                    continue
                user_excerpt = self._dialogue_meta_user_excerpt(item)
                if not user_excerpt:
                    continue
                excerpt_head = user_excerpt[:24].rstrip("，,；;。 ")
                user_label = self.locale.text("dialogue_meta", "user_label", "用户")
                frame_markers = self.locale.list("dialogue_meta", "required_frame_markers")
                has_dialogue_frame = user_label in merged_text and any(
                    marker in merged_text for marker in frame_markers
                )
                if not has_dialogue_frame or (excerpt_head and excerpt_head not in merged_text):
                    return ["dialogue_meta"]
        if block.block_type == "topic" and any(self._conversation_memory_query_intent(item) for item in evidence):
            memory_query_markers = self.locale.list("memory_query", "required_render_markers")
            if not any(marker in merged_text for marker in memory_query_markers):
                return ["conversation_memory_query"]
        return [
            term
            for term in block.protected_terms
            if self._term_matches(term, evidence_text) and not self._term_matches(term, merged_text)
        ]

    @staticmethod
    def _locked_update_allowed(block: CompressionBlock, evidence: list[str]) -> bool:
        if block.id not in {"project_mainline", "runtime_config"}:
            return True
        evidence_text = "\n".join(evidence)
        negative_markers = [
            "没有确认",
            "未确认",
            "不是确认",
            "不能重写",
            "不能写入",
            "不应重写",
            "不应写入",
            "保持不变",
            "仍以 .env 为准",
        ]
        if any(marker in evidence_text for marker in negative_markers):
            return False
        if block.id == "project_mainline":
            update_markers = ["项目主线更新", "主线更新", "目标更新", "项目目标更新", "主线改为", "目标改为"]
        else:
            update_markers = ["确认更新", "已确认更新", "正式更新", "更新为", "改为", "设为", "设置为", "正式配置"]
        return any(marker in evidence_text for marker in update_markers)

    @staticmethod
    def _is_unconfirmed_locked_update(block: CompressionBlock, fact: str) -> bool:
        if block.id not in {"project_mainline", "runtime_config"}:
            return False
        unconfirmed_markers = [
            "有人",
            "随口",
            "提议",
            "未确认",
            "不是确认",
            "不能写入",
            "反例",
            "测试假设",
            "仍以 .env 为准",
        ]
        return any(marker in fact for marker in unconfirmed_markers)

    def _mark_dependents_stale(self, block_id: str) -> None:
        for block in self.blocks.values():
            if block_id in block.depends_on:
                block.stale = True

    def _ordered_blocks_for_render(self) -> list[CompressionBlock]:
        stability_rank = {"locked": 0, "stable": 1, "volatile": 2}
        return sorted(
            self.blocks.values(),
            key=lambda block: (stability_rank[block.stability], block.updated_turn, block.id),
        )

    def _render_block(self, block: CompressionBlock) -> str:
        lines = [f"[{block.title}]"]
        for fact in block.facts:
            cleaned = self._clean_fact(fact)
            if cleaned and not self._is_low_value_fact(block, cleaned):
                lines.append(f"- {cleaned}")
        for risk in block.risks:
            cleaned = self._clean_fact(risk)
            if cleaned and not self._is_low_value_fact(block, cleaned):
                lines.append(f"- 风险：{cleaned}")
        if len(lines) == 1:
            return ""
        return "\n".join(lines)

    def _trim_block_to_budget(self, block: CompressionBlock, budget: int) -> str:
        if budget <= 0:
            return ""
        lines = [f"[{block.title}]"]
        for item in [*block.facts, *[f"风险：{risk}" for risk in block.risks]]:
            trial = "\n".join([*lines, f"- {item}"])
            if estimate_text_tokens(trial) > budget:
                break
            lines.append(f"- {item}")
        return "\n".join(lines) if len(lines) > 1 else ""

    def _cache_key(
        self,
        block: CompressionBlock,
        evidence: list[str],
        target_tokens: int,
        semantic_dag: TurnSemanticDAG | None = None,
    ) -> str:
        dag_hash = None
        if semantic_dag is not None:
            dag_hash = _stable_hash(self._compression_dag_material_payload(semantic_dag, block))
        payload = {
            "policy_version": self.policy_version,
            "prompt_version": self.prompt_version,
            "block_id": block.id,
            "old_hash": block.content_hash,
            "delta_hash": _stable_hash(evidence),
            "dag_hash": dag_hash,
            "target_tokens": target_tokens,
            "compression_update_token_budget": self.compression_update_token_budget,
            "compression_update_posterior_budget": self._current_compression_update_budget_tokens(),
            "compression_update_ramp_turns": self.compression_update_ramp_turns,
        }
        return _stable_hash(payload)

    @staticmethod
    def _merge_turns(source_turns: list[int], turn_id: int) -> list[int]:
        turns = [*source_turns, turn_id]
        result: list[int] = []
        for turn in turns:
            if turn not in result:
                result.append(turn)
        return result[-12:]

    @staticmethod
    def _merge_with_limit(existing: list[str], incoming: list[str], limit: int, prefer_existing: bool = True) -> list[str]:
        merged = CompressionPool._dedupe_keep_order([*existing, *incoming])
        if len(merged) <= limit:
            return merged
        if not prefer_existing:
            return merged[-limit:]
        kept: list[str] = []
        for item in existing:
            if item not in kept:
                kept.append(item)
            if len(kept) >= limit:
                return kept
        for item in incoming:
            if item not in kept:
                kept.append(item)
            if len(kept) >= limit:
                return kept
        return kept

    def _merge_topic_facts(self, block: CompressionBlock, incoming: list[str], turn_id: int) -> list[str]:
        prepared = self._prepare_topic_facts(block, incoming, turn_id)
        merged = self._dedupe_keep_order([*block.facts, *prepared])
        return self._compact_topic_facts(merged, block.max_facts)

    def _prepare_topic_facts(self, block: CompressionBlock, incoming: list[str], turn_id: int) -> list[str]:
        existing_core = {self._topic_fact_core(item) for item in block.facts}
        prepared: list[str] = []
        for item in incoming:
            cleaned = self._clean_fact(item)
            if not cleaned or self._is_low_value_fact(block, cleaned):
                continue
            item_turn = self._topic_turn_id(cleaned)
            if item_turn is not None and item_turn != turn_id:
                continue
            core = self._topic_fact_core(cleaned)
            if not core or core in existing_core:
                continue
            core = self._shorten_topic_fact(core)
            if not core:
                continue
            prepared.append(f"第{turn_id}轮：{core}")
            existing_core.add(core)
            if len(prepared) >= CHAT_TOPIC_FACTS_PER_TURN:
                break
        return prepared

    def _compact_topic_facts(self, facts: list[str], limit: int) -> list[str]:
        archive_sources: list[str] = []
        normal_facts: list[str] = []
        for fact in self._dedupe_keep_order(facts):
            if fact.startswith("早期摘要："):
                archive_sources.append(fact.removeprefix("早期摘要：").strip())
            else:
                normal_facts.append(fact)

        turn_ids = sorted({turn_id for fact in normal_facts if (turn_id := self._topic_turn_id(fact)) is not None})
        recent_turns = set(turn_ids[-CHAT_TOPIC_RECENT_TURNS:])
        recent_facts: list[str] = []
        for fact in normal_facts:
            turn_id = self._topic_turn_id(fact)
            if turn_id is None or turn_id in recent_turns:
                recent_facts.append(fact)
            else:
                archive_sources.append(fact)

        result: list[str] = []
        archive = self._build_topic_archive(archive_sources)
        if archive:
            result.append(archive)
        result.extend(recent_facts)

        if len(result) <= limit:
            return result
        if archive:
            return [archive, *result[-(limit - 1) :]]
        overflow_count = max(1, len(result) - limit + 1)
        overflow_archive = self._build_topic_archive(result[:overflow_count])
        if overflow_archive:
            remaining = result[overflow_count:]
            return [overflow_archive, *remaining[-(limit - 1) :]]
        return result[-limit:]

    def _build_topic_archive(self, sources: list[str]) -> str:
        cores: list[str] = []
        for source in sources:
            core = self._topic_fact_core(source)
            if core and not any(core == existing for existing in cores):
                cores.append(core)
        if not cores:
            return ""
        archive = "；".join(cores)
        if len(archive) > CHAT_TOPIC_ARCHIVE_MAX_CHARS:
            archive = archive[: CHAT_TOPIC_ARCHIVE_MAX_CHARS - 1].rstrip("；,，。 ") + "…"
        return f"早期摘要：{archive}"

    @staticmethod
    def _topic_turn_id(fact: str) -> int | None:
        match = re.match(r"^第(\d+)轮[：:]", _normalize_space(fact))
        return int(match.group(1)) if match else None

    @staticmethod
    def _topic_fact_core(fact: str) -> str:
        core = re.sub(r"^第\d+轮[：:]?\s*", "", _normalize_space(fact))
        core = core.removeprefix("早期摘要：").strip()
        return core

    @staticmethod
    def _shorten_topic_fact(fact: str) -> str:
        fact = _normalize_space(fact)
        if len(fact) <= CHAT_TOPIC_FACT_MAX_CHARS:
            return fact
        return fact[: CHAT_TOPIC_FACT_MAX_CHARS - 1].rstrip("，,；;。 ") + "…"

    def _extract_keywords(self, text: str, max_keywords: int = 12) -> list[str]:
        normalized = _normalize_space(text)
        if not normalized:
            return []

        tokens: list[str] = []
        tokens.extend(re.findall(r"\b[a-zA-Z][a-zA-Z0-9_./%-]{1,}\b", normalized))
        if self._english_nlp is not None:
            tokens.extend(self._english_nlp.keyword_candidates(normalized))
        for part in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
            if len(part) <= 8:
                tokens.append(part)
                continue
            for fragment in re.split(r"(?:需要|必须|不能|可以|以及|然后|因为|所以|对于|关于|通过|采用|实现)", part):
                fragment = fragment.strip()
                if 2 <= len(fragment) <= 8:
                    tokens.append(fragment)
            tokens.append(part[:8])

        stopwords = {word.lower() for word in self.locale.list("keyword_extraction", "stopwords")}
        filtered = [
            token
            for token in tokens
            if len(token.strip()) >= 2 and token.lower() not in stopwords and token not in stopwords
        ]
        return CompressionPool._dedupe_keep_order(filtered)[:max_keywords]

    @staticmethod
    def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            key = _normalize_space(item)
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(key)
        return result

    @staticmethod
    def _has_cjk(text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    @staticmethod
    def _word_variants(word: str) -> set[str]:
        lowered = word.lower().strip("'’")
        variants = {lowered}
        irregular = {
            "ate": "eat",
            "eaten": "eat",
            "ran": "run",
            "gone": "go",
            "went": "go",
            "bought": "buy",
            "brought": "bring",
            "kept": "keep",
            "made": "make",
            "said": "say",
            "thought": "think",
            "children": "child",
            "people": "person",
            "men": "man",
            "women": "woman",
            "better": "good",
            "best": "good",
            "worse": "bad",
            "worst": "bad",
        }
        if lowered in irregular:
            variants.add(irregular[lowered])
        if lowered.endswith("'s"):
            variants.add(lowered[:-2])
        if len(lowered) > 4 and lowered.endswith("ies"):
            variants.add(lowered[:-3] + "y")
        for suffix in ("ing", "ed", "es", "s"):
            if len(lowered) <= len(suffix) + 2 or not lowered.endswith(suffix):
                continue
            base = lowered[: -len(suffix)]
            variants.add(base)
            if suffix in {"ing", "ed", "es"}:
                variants.add(base + "e")
            if suffix in {"ing", "ed"}:
                if len(base) >= 2 and base[-1] == base[-2]:
                    variants.add(base[:-1])
        return {variant for variant in variants if variant}

    def _term_matches(self, term: str, text: str) -> bool:
        normalized_term = _normalize_space(term)
        normalized_text = _normalize_space(text)
        if not normalized_term or not normalized_text:
            return False
        if self._has_cjk(normalized_term):
            return normalized_term in normalized_text
        term_lower = normalized_term.lower()
        text_lower = normalized_text.lower()
        symbolic_term = bool(re.search(r"[^a-z'’\s-]", term_lower))
        if symbolic_term:
            return term_lower in text_lower
        if self._english_nlp is not None and re.search(r"[a-z]", term_lower):
            return self._english_nlp.term_matches(term_lower, text_lower)
        if " " in term_lower or "-" in term_lower:
            return term_lower in text_lower
        term_variants = self._word_variants(term_lower)
        for token in re.findall(r"\b[a-zA-Z][a-zA-Z'’-]*\b", text_lower):
            if term_variants & self._word_variants(token):
                return True
        return term_lower in text_lower

    @staticmethod
    def _clean_fact(text: str) -> str:
        text = _normalize_space(text)
        text = re.sub(r"^(用户|助手|user|assistant)[:：]\s*", "", text, flags=re.I)
        text = re.sub(r"请只回复[:：]?.*$", "", text).strip()
        text = re.sub(r"^#{1,6}\s*", "", text)
        text = re.sub(r"^\d+[.、]\s*", "", text)
        text = text.replace("**", "").replace("`", "")
        text = re.sub(r"^(?:(?:要点|风险|事实|摘要)[:：]\s*)+", "", text)
        text = text.strip("-•* \t")
        if len(text) > 120:
            text = text[:117].rstrip() + "..."
        return text

    def _is_low_value_fact(self, block: CompressionBlock, fact: str) -> bool:
        normalized = _normalize_space(fact)
        if not normalized:
            return True
        normalized_heading = normalized.rstrip("。；;:：")
        prompt_like_fragments = self.locale.list("low_value", "prompt_like_fragments")
        if any(fragment and (fragment in normalized or fragment.lower() in normalized.lower()) for fragment in prompt_like_fragments):
            return True
        if normalized.startswith("def ") or normalized.startswith("class "):
            return True
        if block.block_type != "config" and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", normalized):
            return True
        if re.search(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\(", normalized):
            return True
        preference_terms = self.locale.list("low_value", "non_preference_block_preference_terms")
        if block.id != "user_preference" and any(self._term_matches(term, normalized) for term in preference_terms):
            return True
        low_value_headings = set(self.locale.list("low_value", "headings"))
        if normalized_heading in low_value_headings:
            return True
        if block.block_type == "risk" and any(fragment in normalized for fragment in ["是指攻击者", "实施策略与逻辑框架"]):
            return True
        if block.block_type != "topic" and re.match(r"^第\d+轮[：:]", normalized):
            return True
        known_terms = {term.strip("`* ") for term in [*block.keywords, *block.protected_terms] if term}
        if block.block_type != "config" and normalized in known_terms and normalized in "\n".join([*block.facts, *block.risks]):
            return True
        return False

    @staticmethod
    def _shares_meaningful_fragment(left: str, right: str) -> bool:
        left_terms = {term for term in re.split(r"[^\w\u4e00-\u9fff]+", left) if len(term) >= 3}
        right_terms = {term for term in re.split(r"[^\w\u4e00-\u9fff]+", right) if len(term) >= 3}
        return bool(left_terms & right_terms)


def make_fake_llm_response(text: str, input_tokens: int = 400, output_tokens: int = 80) -> LLMResponse:
    return LLMResponse(
        text=text,
        stats=LLMRunStats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            wall_seconds=input_tokens / 120.0 + output_tokens / 30.0,
            prefill_seconds=input_tokens / 120.0,
            decode_seconds=output_tokens / 30.0,
            peak_memory_gb=7.4,
        ),
    )

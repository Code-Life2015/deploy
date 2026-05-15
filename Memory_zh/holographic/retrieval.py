"""Hybrid keyword/BM25 retrieval for the memory store.

Ported from KIK memory_agent.py — combines FTS5 full-text search with
Jaccard similarity reranking, BM25 ranking, and trust-weighted scoring.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np  # noqa: F401 — used in type comments only

if TYPE_CHECKING:
    from .store import MemoryStore

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]


# ─── Query Classification (merged intent + routing) ───────────────────────────

# Intent detection: strong-first keyword → intent label
_INTENT_KEYWORDS: dict[str, str] = {
    # Update / edit existing facts
    "修改": "update", "更新": "update", "更正": "update", "纠正": "update",
    # Delete / remove facts
    "删除": "delete", "移除": "delete", "去掉": "delete",
    # Add / store new facts
    "添加": "store", "新增": "store", "记住": "store", "记录": "store",
    "保存": "store", "加": "store",
    # Delete + store together = replace
    "替换": "replace", "覆盖": "replace",
}

# Query-type routing: strong-first keyword → (method_name, is_likes_fallback)
_QUERY_TYPE_KEYWORDS: dict[str, tuple[str, bool]] = {
    # probe — HRR structural query (unbind/extract)
    "探测": ("probe", False), "probe": ("probe", False),
    "谁记": ("probe", False), "哪些事实": ("probe", False),
    # reason — multi-entity compositional query
    "推理": ("reason", False), "reason": ("reason", False),
    "为什么": ("reason", False), "怎么关联": ("reason", False),
    # contradict — contradiction detection
    "矛盾": ("contradict", False), "冲突": ("contradict", False),
    "contradict": ("contradict", False),
    # contradict + search — cross-reference with contradiction check
    "验证": ("contradict", False), "核实": ("contradict", False),
}

# Keyword → category routing map
# Maps query keywords → fact.category values for filtered search.
# Only use this when you KNOW the DB categories match the keywords.
_CATEGORY_KEYWORDS: dict[str, str] = {
    "用户": "user_pref", "偏好": "user_pref", "喜欢": "user_pref",
    "记忆": "memory", "记忆系统": "memory", "holographic": "memory",
    "编码": "code", "代码": "code", "bug": "code", "函数": "code",
    "实现": "code", "修复": "code",
    "安全": "security", "security": "security",
    "配置": "config", "config": "config", "设置": "config",
    "项目": "project", "project": "project",
    "测试": "testing", "testing": "testing",
    # NOTE: "验证" intentionally omitted — it is a method keyword (contradict),
    # not a category keyword. Keep them strictly orthogonal.
    "性能": "performance", "performance": "performance", "优化": "performance",
}


@dataclass(slots=True)
class QueryClassification:
    """Single-pass query analysis result.

    Attributes
    ----------
    intent : str
        User intent: "retrieve" (default), "store", "update", "delete",
        "replace", or "search" (neutral search).
    method : str
        Retrieval method to call: "search", "probe", "reason", "contradict",
        "related", or "add" / "update" / "remove".
    query_type : str
        Routing decision: "fts5", "likes", or "mixed".
    category : str | None
        Auto-detected category (None = no filter).
    cleaned_query : str
        FTS5-normalized query string (safe for FTS5 MATCH, even when using LIKE).
    needs_likes : bool
        True → skip FTS5, caller must use LIKE-based candidates method.
    reason : str
        Human-readable explanation of routing decisions.
    """
    intent: str
    method: str
    query_type: str
    category: str | None
    cleaned_query: str
    needs_likes: bool
    reason: str


def classify_query(query: str) -> QueryClassification:
    """Single-pass intent detection + query-type routing + category mapping.

    Merges ``_preprocess_fts5_query`` and ``_auto_category`` into one
    function that returns everything needed to dispatch to the correct
    handler without re-parsing the query.

    Parameters
    ----------
    query : str
        Raw user query string.

    Returns
    -------
    QueryClassification
        Dataclass with intent, method, query_type, category,
        cleaned_query, needs_likes, and reason.
    """
    original = query.strip()
    lower = original.lower()

    # ── Step 1: Intent detection ────────────────────────────────────────────
    for keyword, intent in _INTENT_KEYWORDS.items():
        if keyword in lower:
            resolved_intent = intent
            break
    else:
        resolved_intent = "retrieve"

    # ── Step 2: Query-type routing ───────────────────────────────────────
    for keyword, (method, likes) in _QUERY_TYPE_KEYWORDS.items():
        if keyword in lower:
            resolved_method = method
            needs_likes = likes
            # reason handled below after classification
            break
    else:
        resolved_method = resolved_intent  # default: intent = method for non-search
        needs_likes = False

    # Override method for neutral search intent only.
    # If a method keyword was already detected (probe/reason/contradict)
    # or an intent keyword (update/delete/store/replace), keep it.
    if resolved_intent == "retrieve" and resolved_method == "retrieve":
        resolved_method = "search"

    # ── Step 3: FTS5 query normalization ──────────────────────────────────
    # Only apply FTS5 preprocessing for search method.
    # Intent methods (probe/reason/contradict) handle their own routing.
    if resolved_method == "search":
        cleaned, _needs_likes_fts = _preprocess_fts5_query(original)
        needs_likes = needs_likes or _needs_likes_fts
    else:
        cleaned = original

    # ── Step 4: Category mapping ──────────────────────────────────────────
    # Always run category mapping — category keywords (安全/用户/编码) are
    # orthogonal to intent keywords (修改/删除/添加). When both appear in
    # the same query, return both so the caller can handle the composition
    # (e.g. "修改安全配置" → intent=update, cat=security).
    category = None
    for keyword, cat in _CATEGORY_KEYWORDS.items():
        if keyword in lower:
            category = cat
            break

    # ── Step 5: Reason string ─────────────────────────────────────────────
    reasons: list[str] = []
    if resolved_intent != "retrieve":
        reasons.append(f"intent={resolved_intent}")
    if category:
        reasons.append(f"cat={category}")
    if needs_likes:
        reasons.append("LIKE-fallback")
    else:
        reasons.append("FTS5")
    if resolved_method != "search":
        reasons.append(f"method={resolved_method}")
    reason_str = "; ".join(reasons) if reasons else "default"

    return QueryClassification(
        intent=resolved_intent,
        method=resolved_method,
        query_type="likes" if needs_likes else "fts5",
        category=category,
        cleaned_query=cleaned,
        needs_likes=needs_likes,
        reason=reason_str,
    )
def _auto_category(query: str, store: MemoryStore) -> str | None:
    """Guess category from query keywords via DB categories table."""
    q = query.lower()
    cat_map = store._load_categories_from_db()
    for keyword, category in cat_map.items():
        if keyword.lower() in q:
            return category
    return None
_fts_special_chars = re.compile(r'[(){}[\\]"\']')

# Compiled once
_re_cjk_char = re.compile(r'[\u4e00-\u9fff]')


def _preprocess_fts5_query(query: str) -> tuple[str, bool]:
    """Normalize user query for FTS5 MATCH.

    Returns (processed_query, is_fallback_likes). When is_fallback_likes=True,
    the caller should fall back to LIKE matching because the query contains
    single Chinese characters that unicode61 treats as stop words, or mixed
    Chinese+English terms.

    Handles:
    - Chinese characters (pass through as-is, unicode61 tokenizer handles them)
    - English terms with special chars (strip chars, keep word)
    - Single Chinese chars → flagged for LIKE fallback
    - Mixed Chinese+English queries → ALWAYS use LIKE (FTS5 BM25 ignores Chinese)
    - Multiple whitespace → single space
    - Empty → return wildcard
    """
    query = query.strip()
    if not query:
        return ('"*', False)

    # Strip FTS5 special punctuation, keep alphanumerics and spaces
    # Chinese chars pass through unchanged
    cleaned = _fts_special_chars.sub(' ', query)
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    if not cleaned:
        return ('"*', False)

    # Detect single Chinese characters (unicode range CJK Unified Ideographs)
    # FTS5 unicode61 treats them as stop words → need LIKE fallback
    cjk_chars = _re_cjk_char.findall(cleaned)
    # CJK tokens: ALWAYS use LIKE fallback.
    # FTS5 unicode61 word-boundary tokenizes Chinese as individual chars,
    # so "量化" (2 chars) would be indexed as two separate stop words → zero hits.
    # LIKE %term% does character-level substring match and works reliably.
    if cjk_chars:
        needs_likes = True
    # Short single-term English queries → LIKE (FTS5 BM25 misbehaves on rare/short terms)
    elif is_single_term:
        needs_likes = True

    return cleaned, needs_likes


def _auto_category(query: str) -> str | None:
    """Guess category from query keywords (legacy standalone variant)."""
    return None  # now delegated to FactRetriever._auto_category via store


class FactRetriever:
    """Multi-strategy fact retrieval with trust-weighted scoring."""

    def __init__(
        self,
        store: MemoryStore,
        temporal_decay_half_life: int = 0,  # days, 0 = disabled
        fts_weight: float = 0.5,
        jaccard_weight: float = 0.3,
        hrr_weight: float = 0.2,
        hrr_dim: int = 1024,
    ):
        self.store = store
        self.half_life = temporal_decay_half_life
        self.hrr_dim = hrr_dim

        # Auto-redistribute weights if numpy unavailable
        if hrr_weight > 0 and not hrr._HAS_NUMPY:
            fts_weight = 0.6
            jaccard_weight = 0.4
            hrr_weight = 0.0

        self.fts_weight = fts_weight
        self.jaccard_weight = jaccard_weight
        self.hrr_weight = hrr_weight

    def _auto_category(self, query: str) -> str | None:
        """Guess category from query using DB categories table (lazy-loaded)."""
        q = query.lower()
        cat_map = self.store._load_categories_from_db()
        for keyword, cat in cat_map.items():
            if keyword.lower() in q:
                return cat
        return None

    def search(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Hybrid search: FTS5 candidates → Jaccard rerank → trust weighting.

        Pipeline:
        1. Query preprocessing: escape FTS5 specials + auto-detect category
        2. FTS5 search: Get limit*3 candidates (BM25 ranked)
        3. Jaccard rerank: token overlap between query and fact content
        4. Trust weighting: final_score = relevance * trust_score
        5. Temporal decay (optional): decay = 0.5^(age_days / half_life)
        6. retrieval_count updated for every returned fact

        Returns list of dicts with fact data + 'score' field, sorted by score desc.
        """
        # Stage 0: Query preprocessing + auto category routing
        fts_query, needs_likes = _preprocess_fts5_query(query)
        effective_category = category or self._auto_category(query)

        # Stage 1: Get FTS5 candidates (more than limit for reranking headroom)
        # If needs_likes=True, skip FTS5 and go directly to LIKE fallback
        if needs_likes:
            candidates = self._likes_candidates(query, effective_category, min_trust, limit * 3)
        else:
            candidates = self._fts_candidates(fts_query, effective_category, min_trust, limit * 3)

        # If category routing produced no results, retry without category filter
        if not candidates and effective_category is not None:
            if needs_likes:
                candidates = self._likes_candidates(query, None, min_trust, limit * 3)
            else:
                candidates = self._fts_candidates(fts_query, None, min_trust, limit * 3)

        if not candidates:
            return []

        # Stage 2: Rerank with Jaccard + trust + optional decay
        query_tokens = self._tokenize(query)
        scored = []

        for fact in candidates:
            content_tokens = self._tokenize(fact["content"])
            tag_tokens = self._tokenize(fact.get("tags", ""))
            all_tokens = content_tokens | tag_tokens

            jaccard = self._jaccard_similarity(query_tokens, all_tokens)
            fts_score = fact.get("fts_rank", 0.0)

            # HRR similarity
            if self.hrr_weight > 0 and fact.get("hrr_vector"):
                fact_vec = hrr.bytes_to_phases(fact["hrr_vector"])
                query_vec = hrr.encode_text(query, self.hrr_dim)
                hrr_sim = (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0  # shift to [0,1]
            else:
                hrr_sim = 0.5  # neutral

            # Combine FTS5 + Jaccard + HRR
            relevance = (self.fts_weight * fts_score
                        + self.jaccard_weight * jaccard
                        + self.hrr_weight * hrr_sim)

            # Trust weighting
            score = relevance * fact["trust_score"]

            # Optional temporal decay
            if self.half_life > 0:
                score *= self._temporal_decay(fact.get("updated_at") or fact.get("created_at"))

            fact["score"] = score
            scored.append(fact)

        # Sort by score descending, return top limit
        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]
        # Strip raw HRR bytes — callers expect JSON-serializable dicts
        for fact in results:
            fact.pop("hrr_vector", None)

        # Update retrieval_count for all returned facts (key fix)
        if results and self.store:
            ids = [r["fact_id"] for r in results]
            placeholders = ",".join("?" * len(ids))
            self.store._conn.execute(
                f"UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id IN ({placeholders})",
                ids,
            )
            self.store._conn.commit()

        return results

    def probe(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Compositional entity query using HRR algebra.

        Unbinds entity from memory bank to extract associated content.
        This is NOT keyword search — it uses algebraic structure to find facts
        where the entity plays a structural role.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            # Fallback to keyword search on entity name
            return self.search(entity, category=category, limit=limit)

        conn = self.store._conn

        # Encode entity as role-bound vector
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
        entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)
        probe_key = hrr.bind(entity_vec, role_entity)

        # Try category-specific bank first, then all facts
        if category:
            bank_name = f"cat:{category}"
            bank_row = conn.execute(
                "SELECT vector FROM memory_banks WHERE bank_name = ?",
                (bank_name,),
            ).fetchone()
            if bank_row:
                bank_vec = hrr.bytes_to_phases(bank_row["vector"])
                extracted = hrr.unbind(bank_vec, probe_key)
                # Use extracted signal to score individual facts
                return self._score_facts_by_vector(
                    extracted, category=category, limit=limit
                )

        # Score against individual fact vectors directly
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            # Final fallback: keyword search
            return self.search(entity, category=category, limit=limit)

        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))
            # Unbind probe key from fact to see if entity is structurally present
            residual = hrr.unbind(fact_vec, probe_key)
            # Compare residual against content signal
            role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)
            content_vec = hrr.bind(hrr.encode_text(fact["content"], self.hrr_dim), role_content)
            sim = hrr.similarity(residual, content_vec)
            fact["score"] = (sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def related(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Discover facts that share structural connections with an entity.

        Unlike probe (which finds facts *about* an entity), related finds
        facts that are connected through shared context — e.g., other entities
        mentioned alongside this one, or content that overlaps structurally.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            return self.search(entity, category=category, limit=limit)

        conn = self.store._conn

        # Encode entity as a bare atom (not role-bound — we want ANY structural match)
        entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)

        # Get all facts with vectors
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            return self.search(entity, category=category, limit=limit)

        # Score each fact by how much the entity's atom appears in its vector
        # This catches both role-bound entity matches AND content word matches
        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))

            # Check structural similarity: unbind entity from fact
            residual = hrr.unbind(fact_vec, entity_vec)
            # A high-similarity residual to ANY known role vector means this entity
            # plays a structural role in the fact
            role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
            role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)

            entity_role_sim = hrr.similarity(residual, role_entity)
            content_role_sim = hrr.similarity(residual, role_content)
            # Take the max — entity could appear in either role
            best_sim = max(entity_role_sim, content_role_sim)

            fact["score"] = (best_sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def reason(
        self,
        entities: list[str],
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Multi-entity compositional query — vector-space JOIN.

        Given multiple entities, algebraically intersects their structural
        connections to find facts related to ALL of them simultaneously.
        This is compositional reasoning that no embedding DB can do.

        Example: reason(["peppi", "backend"]) finds facts where peppi AND
        backend both play structural roles — without keyword matching.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY or not entities:
            # Fallback: search with all entities as keywords
            query = " ".join(entities)
            return self.search(query, category=category, limit=limit)

        conn = self.store._conn
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)

        # For each entity, compute what the bank "remembers" about it
        # by unbinding entity+role from each fact vector
        entity_residuals = []
        for entity in entities:
            entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)
            probe_key = hrr.bind(entity_vec, role_entity)
            entity_residuals.append(probe_key)

        # Get all facts with vectors
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            query = " ".join(entities)
            return self.search(query, category=category, limit=limit)

        # Score each fact by how much EACH entity is structurally present.
        # A fact scores high only if ALL entities have structural presence
        # (AND semantics via min, vs OR which would use mean/max).
        role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)

        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))

            entity_scores = []
            for probe_key in entity_residuals:
                residual = hrr.unbind(fact_vec, probe_key)
                sim = hrr.similarity(residual, role_content)
                entity_scores.append(sim)

            min_sim = min(entity_scores)
            fact["score"] = (min_sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def contradict(
        self,
        category: str | None = None,
        threshold: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Find potentially contradictory facts via entity overlap + content divergence.

        Two facts contradict when they share entities (same subject) but have
        low content-vector similarity (different claims). This is automated
        memory hygiene — no other memory system does this.

        Returns pairs of facts with a contradiction score.
        Falls back to empty list if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            return []

        conn = self.store._conn

        # Get all facts with vectors and their linked entities
        where = "WHERE f.hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND f.category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                   f.created_at, f.updated_at, f.hrr_vector
            FROM facts f
            {where}
            """,
            params,
        ).fetchall()

        if len(rows) < 2:
            return []

        # Guard against O(n²) explosion on large fact stores.
        # At 500 facts, that's ~125K comparisons — acceptable.
        # Above that, only check the most recently updated facts.
        _MAX_CONTRADICT_FACTS = 500
        if len(rows) > _MAX_CONTRADICT_FACTS:
            rows = sorted(rows, key=lambda r: r["updated_at"] or r["created_at"], reverse=True)
            rows = rows[:_MAX_CONTRADICT_FACTS]

        # Build entity sets per fact
        fact_entities: dict[int, set[str]] = {}
        for row in rows:
            fid = row["fact_id"]
            entity_rows = conn.execute(
                """
                SELECT e.name FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE fe.fact_id = ?
                """,
                (fid,),
            ).fetchall()
            fact_entities[fid] = {r["name"].lower() for r in entity_rows}

        # Compare all pairs: high entity overlap + low content similarity = contradiction
        facts = [dict(r) for r in rows]
        contradictions = []

        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                f1, f2 = facts[i], facts[j]
                ents1 = fact_entities.get(f1["fact_id"], set())
                ents2 = fact_entities.get(f2["fact_id"], set())

                if not ents1 or not ents2:
                    continue

                # Entity overlap (Jaccard)
                entity_overlap = len(ents1 & ents2) / len(ents1 | ents2) if (ents1 | ents2) else 0.0

                if entity_overlap < 0.3:
                    continue  # Not enough entity overlap to be contradictory

                # Content similarity via HRR vectors
                v1 = hrr.bytes_to_phases(f1["hrr_vector"])
                v2 = hrr.bytes_to_phases(f2["hrr_vector"])
                content_sim = hrr.similarity(v1, v2)

                # High entity overlap + low content similarity = potential contradiction
                # contradiction_score: higher = more contradictory
                contradiction_score = entity_overlap * (1.0 - (content_sim + 1.0) / 2.0)

                if contradiction_score >= threshold:
                    # Strip hrr_vector from output (not JSON serializable)
                    f1_clean = {k: v for k, v in f1.items() if k != "hrr_vector"}
                    f2_clean = {k: v for k, v in f2.items() if k != "hrr_vector"}
                    contradictions.append({
                        "fact_a": f1_clean,
                        "fact_b": f2_clean,
                        "entity_overlap": round(entity_overlap, 3),
                        "content_similarity": round(content_sim, 3),
                        "contradiction_score": round(contradiction_score, 3),
                        "shared_entities": sorted(ents1 & ents2),
                    })

        contradictions.sort(key=lambda x: x["contradiction_score"], reverse=True)
        return contradictions[:limit]

    def _score_facts_by_vector(
        self,
        target_vec: "np.ndarray",
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Score facts by similarity to a target vector."""
        conn = self.store._conn

        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))
            sim = hrr.similarity(target_vec, fact_vec)
            fact["score"] = (sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def _fts_candidates(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list[dict]:
        """Get raw FTS5 candidates from the store.

        Uses the store's database connection directly for FTS5 MATCH
        with rank scoring. Normalizes FTS5 rank to [0, 1] range.
        """
        conn = self.store._conn

        # Build query - FTS5 rank is negative (lower = better match)
        # We need to join facts_fts with facts to get all columns
        params: list = []
        where_clauses = ["facts_fts MATCH ?"]
        params.append(query)

        if category:
            where_clauses.append("f.category = ?")
            params.append(category)

        where_clauses.append("f.trust_score >= ?")
        params.append(min_trust)

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT f.*, facts_fts.rank as fts_rank_raw
            FROM facts_fts
            JOIN facts f ON f.fact_id = facts_fts.rowid
            WHERE {where_sql}
            ORDER BY facts_fts.rank
            LIMIT ?
        """
        params.append(limit)

        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            # FTS5 MATCH failed — let caller decide fallback strategy
            return []

        if not rows:
            return []

        # Normalize FTS5 rank: BM25 rank is negative (more negative = better match)
        # Convert to positive score in [0, 1] range where 1 = best match
        raw_ranks = [abs(row["fts_rank_raw"]) for row in rows]
        max_rank = max(raw_ranks) if raw_ranks else 1.0
        max_rank = max(max_rank, 1e-6)  # avoid div by zero

        results = []
        for row, raw_rank in zip(rows, raw_ranks):
            fact = dict(row)
            fact.pop("fts_rank_raw", None)
            fact["fts_rank"] = raw_rank / max_rank  # normalize to [0, 1]
            results.append(fact)

        return results

    @staticmethod
    def _escape_like(token: str) -> str:
        """Escape LIKE wildcard characters to treat them as literals.

        Order matters: backslash MUST be escaped first to avoid double-escaping.
        """
        return token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def _likes_candidates(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list[dict]:
        """LIKE-based fallback for queries with Chinese characters or mixed terms.

        Uses per-token LIKE clauses (OR'd) so that each token is matched
        independently — fixing the previous bug where the whole query string
        was wrapped in a single LIKE pattern that could not match anything.

        Escapes LIKE wildcards in tokens to prevent SQL injection.
        Falls back to empty list on error.
        """
        conn = self.store._conn

        # Split on whitespace — Chinese tokens are preserved as whole words
        tokens = [t.strip() for t in query.split() if t.strip()]
        if not tokens:
            return []

        # Build per-token LIKE clauses with proper escaping
        like_clauses: list[str] = []
        params: list = []
        for tok in tokens:
            esc = self._escape_like(tok)
            like_clauses.append(
                "(f.content LIKE ? ESCAPE '\\' OR f.tags LIKE ? ESCAPE '\\')"
            )
            params.append(f"%{esc}%")
            params.append(f"%{esc}%")

        where_sql = " OR ".join(like_clauses)

        # Assemble WHERE with category and trust filters
        where_parts = [f"({where_sql})", "f.trust_score >= ?"]
        params.append(min_trust)
        if category:
            where_parts.append("f.category = ?")
            params.append(category)

        final_where = " AND ".join(where_parts)
        sql = f"""
            SELECT f.*, 0.5 as fts_rank
            FROM facts f
            WHERE {final_where}
            ORDER BY f.trust_score DESC
            LIMIT ?
        """
        params.append(limit)

        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            return []

        return [dict(row) for row in rows]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple whitespace tokenization with lowercasing.

        Strips common punctuation. No stemming/lemmatization (Phase 1).
        """
        if not text:
            return set()
        # Split on whitespace, lowercase, strip punctuation
        tokens = set()
        for word in text.lower().split():
            cleaned = word.strip(".,;:!?\"'()[]{}#@<>")
            if cleaned:
                tokens.add(cleaned)
        return tokens

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _temporal_decay(self, timestamp_str: str | None) -> float:
        """Exponential decay: 0.5^(age_days / half_life_days).

        Returns 1.0 if decay is disabled or timestamp is missing.
        """
        if not self.half_life or not timestamp_str:
            return 1.0

        try:
            if isinstance(timestamp_str, str):
                # Parse ISO format timestamp from SQLite
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                ts = timestamp_str

            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
            if age_days < 0:
                return 1.0

            return math.pow(0.5, age_days / self.half_life)
        except (ValueError, TypeError):
            return 1.0

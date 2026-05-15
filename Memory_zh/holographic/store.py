"""
SQLite-backed fact store with entity resolution and trust scoring.
Single-user Hermes memory store plugin.
"""

import re
import sqlite3
import threading
from pathlib import Path

import jieba

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER REFERENCES facts(fact_id),
    entity_id INTEGER REFERENCES entities(entity_id),
    PRIMARY KEY (fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_entities_name  ON entities(name);

CREATE TABLE IF NOT EXISTS categories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    keywords    TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS relationships (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source_entity       TEXT NOT NULL,
    target_entity       TEXT NOT NULL,
    relationship_type   TEXT NOT NULL,
    properties          TEXT DEFAULT '{}',
    source_fact_id      INTEGER REFERENCES facts(fact_id),
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_entity);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_entity);
CREATE INDEX IF NOT EXISTS idx_rel_type   ON relationships(relationship_type);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Trust adjustment constants
_HELPFUL_DELTA   =  0.05
_UNHELPFUL_DELTA = -0.10
_TRUST_MIN       =  0.0
_TRUST_MAX       =  1.0

# Entity extraction patterns
_RE_CAPITALIZED      = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
_RE_SINGLE_CAPITALIZED = re.compile(r'\b([A-Z][a-z]{2,}s?)\b')
_RE_DOUBLE_QUOTE      = re.compile(r'"([^"]+)"')
_RE_SINGLE_QUOTE      = re.compile(r"'([^']+)'")
# AKA: group2 = (\w+?\s+\w+(?:\s+\w+)*) — first \w+? + mandatory \s+
# prevents early stop (must have 2 words with space), (?:)* grabs remaining words.
_RE_AKA = re.compile(
    r"\b(\w+(?:\s+\w+)*)\b\s+(?:aka|also known as)\s+"
    r"(\w+?\s+\w+(?:\s+\w+)*)",
    re.IGNORECASE,
)
# Stop words to exclude from single-word entity extraction
_STOP_WORDS = frozenset({
    "the", "this", "that", "these", "those", "it", "its",
    "he", "she", "they", "them", "his", "her", "their",
    "we", "you", "your", "yours", "me", "my", "mine",
    "but", "and", "or", "nor", "so", "yet", "for", "not",
    "from", "with", "within", "without", "against", "among",
    "about", "after", "before", "between", "through", "during",
    "over", "under", "above", "below", "since", "while",
    "although", "because", "if", "else", "when", "where", "who", "whom",
    "what", "why", "how", "then", "there", "here", "once", "just",
    "also", "only", "more", "less", "most", "some", "any", "all",
    "each", "every", "both", "few", "such", "same", "other", "another",
    "one", "two", "three", "four", "five", "first", "second", "third",
    "new", "old", "young", "great", "good", "best", "better", "big",
    "small", "large", "long", "short", "high", "low", "right", "left",
    "next", "last", "last", "early", "late", "back", "way", "well",
    "can", "could", "will", "would", "shall", "should", "may", "might",
    "must", "need", "needs", "ought", "used",
    "have", "has", "had", "having", "do", "does", "did", "doing", "done",
    "was", "were", "are", "is", "been", "being",
    "see", "saw", "seen", "know", "knew", "known",
    "get", "got", "getting", "make", "made", "making",
    "go", "went", "gone", "going", "come", "came", "coming",
    "take", "took", "taken", "give", "gave", "given",
    "into", "onto", "out", "off", "down", "up", "in", "out",
    "very", "too", "quite", "rather", "almost", "enough",
    "much", "many", "like", "as", "than", "even",
    "now", "still", "already", "always", "never", "ever",
    "really", "truly", "however", "therefore", "thus", "hence",
    "anyway", "besides", "instead", "though", "whether",
    "maybe", "perhaps", "probably", "actually", "certainly", "definitely",
    "exactly", "simply", "merely", "barely", "nearly",
    "else", "someone", "anyone", "everyone", "noone", "nothing",
    "something", "anything", "everything", "anywhere", "everywhere",
    "sql", "api", "http", "https", "tcp", "udp", "url", "uri",
})

# Chinese generic words / 通用词黑名单 — filtered from entity extraction
_CHINESE_GENERIC_WORDS = frozenset({
    # Organization suffixes / 组织后缀
    "公司", "集团", "企业", "机构", "组织", "总部", "分部",
    "有限公司", "股份有限公司", "有限责任公司",
    "工厂", "车间", "事业部", "子公司", "母公司", "控股",
    "公司总部",  # compound: too generic even if not in stop list
    # Role/title suffixes / 职位后缀
    "公司", "部门", "科室", "科室", "小组", "团队", "小组",
    # Location generics / 地点泛称
    "地方", "地区", "区域", "地带", "地段",
    "城市", "城镇", "乡村", "农村", "城市",
    "国家", "国内", "国外", "海外", "境内", "境外",
    "全球", "世界", "国际", "全国", "全", "各地",
    # Abstract/generic terms / 抽象泛称
    "公司", "产品", "业务", "服务", "方案", "技术", "系统", "平台",
    "项目", "工程", "计划", "方案", "战略", "政策", "方针",
    "市场", "行业", "领域", "方向", "方向",
    "研究", "研发", "生产", "制造", "销售", "采购", "供应",
    "管理", "运营", "经营", "运作", "执行", "实施",
    "发展", "成长", "增长", "扩张", "扩展", "扩大",
    "合作", "合作", "协同", "协调", "配合", "协作",
    "主要", "重要", "关键", "核心", "根本", "基本",
    "相关", "有关", "各类", "各种", "一切", "所有",
    "其他", "另外", "此外", "包括", "除外的", "即", "即", "包括",
    # Measure words / 量词 (shouldn't appear alone but just in case)
    "个", "种", "类", "些", "点儿", "系列", "套", "台", "件",
    "次", "步", "期", "阶段", "代", "版",
    # Common verbs / 常见动名 (extracted as noun by jieba)
    "公司", "工作", "处理", "进行", "发生", "出现", "开始", "结束",
    "使用", "采用", "利用", "运用", "应用", "作用", "功能",
    "提供", "支持", "帮助", "解决", "实现", "完成", "达到",
    "认为", "觉得", "知道", "认识", "了解", "理解", "明白",
    "希望", "想要", "需要", "要求", "必须", "应该", "可以",
    # Time/general
    "时间", "时候", "期间", "阶段", "时期", "年代", "岁月",
    "当前", "目前", "现在", "今天", "昨日", "明日", "将来",
    "历史", "传统", "主流", "主要", "基本", "根本",
    # Negation
    "不", "没", "无", "非", "否", "别", "莫",
    # Pronouns/demonstratives
    "这", "那", "此", "各", "每", "某", "任何", "所有",
    # Numerical expressions jieba splits
    "第一", "第二", "第三", "第一", "第二",
    # Domain jargon that is too generic
    "信息", "数据", "网络", "数字", "电子", "智能", "科技",
    "计算", "处理", "分析", "管理", "监控", "监测",
    "安全", "保密", "隐私", "可靠", "高效", "优质", "先进",
    # Common compounds jieba splits that we DON'T want as entities
    "华为",  # will be handled by jieba phrase extraction
})


def _clamp_trust(value: float) -> float:
    return max(_TRUST_MIN, min(_TRUST_MAX, value))


class MemoryStore:
    """SQLite-backed fact store with entity resolution and trust scoring."""

    def __init__(
        self,
        db_path: "str | Path | None" = None,
        default_trust: float = 0.5,
        hrr_dim: int = 1024,
    ) -> None:
        if db_path is None:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "memory_store.db")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_trust = _clamp_trust(default_trust)
        self.hrr_dim = hrr_dim
        self._hrr_available = hrr._HAS_NUMPY
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=10.0,
        )
        self._lock = threading.RLock()
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables, indexes, and triggers if they do not exist. Enable WAL mode."""
        # Use the shared WAL-fallback helper so memory_store.db degrades
        # gracefully on NFS/SMB/FUSE-mounted HERMES_HOME (same issue as
        # state.db / kanban.db — see hermes_state._WAL_INCOMPAT_MARKERS).
        from hermes_state import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="memory_store.db (holographic)")
        self._conn.executescript(_SCHEMA)
        # Seed categories table if empty
        self._conn.execute(
            "INSERT OR IGNORE INTO categories (name, keywords, description) VALUES (?, ?, ?)",
            ["量化交易", "选股,量化,回测,tushare,股价,a股,涨停,macd,换手率,市值,北向,量化选股,量化策略", "量化交易与模型研究相关事实"],
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO categories (name, keywords, description) VALUES (?, ?, ?)",
            ["security", "bandit,安全,漏洞,cve,安全扫描,semgrep,SAST,漏洞扫描", "安全扫描与漏洞检测"],
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO categories (name, keywords, description) VALUES (?, ?, ?)",
            ["system", "git,repo,项目,代码库,仓库,分支,提交", "项目与代码库管理"],
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO categories (name, keywords, description) VALUES (?, ?, ?)",
            ["quant", "因子,多因子,alpha,因子分析,因子挖掘", "量化因子研究"],
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO categories (name, keywords, description) VALUES (?, ?, ?)",
            ["机器学习", "机器学习,ML,machine learning,深度学习,deep learning,神经网络,CNN,RNN", "机器学习相关"],
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO categories (name, keywords, description) VALUES (?, ?, ?)",
            ["人工智能", "人工智能,AI,人工智能,大模型,LLM,AGI,NLP", "人工智能相关"],
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO categories (name, keywords, description) VALUES (?, ?, ?)",
            ["数据分析", "数据分析,数据处理,pandas,数据清洗,ETL,可视化", "数据分析相关"],
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO categories (name, keywords, description) VALUES (?, ?, ?)",
            ["Web开发", "Web开发,前端,后端,API,HTTP,REST,全栈", "Web开发相关"],
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO categories (name, keywords, description) VALUES (?, ?, ?)",
            ["devops", "devops,部署,docker,k8s,kubernetes,CI/CD,自动化", "DevOps与部署"],
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO categories (name, keywords, description) VALUES (?, ?, ?)",
            ["general", "一般,通用,其他", "通用分类"],
        )
        # Migrate: add hrr_vector column if missing (safe for existing databases)
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()}
        if "hrr_vector" not in columns:
            self._conn.execute("ALTER TABLE facts ADD COLUMN hrr_vector BLOB")
        # Rebuild FTS5 index on every init to fix any silent corruption
        try:
            self._conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
        except Exception:
            pass  # FTS5 unavailable — LIKE fallback will still work
        self._conn.commit()

    # ------------------------------------------------------------------
    # Category helpers (lazy-loaded from DB)
    # ------------------------------------------------------------------

    def _load_categories_from_db(self) -> dict[str, str]:
        """Load keyword→category map from DB categories table. Cached."""
        if not hasattr(self, "_categories_cache"):
            rows = self._conn.execute(
                "SELECT name, keywords FROM categories"
            ).fetchall()
            self._categories_cache: dict[str, str] = {}
            for name, keywords in rows:
                for kw in keywords.split(","):
                    kw = kw.strip()
                    if kw:
                        self._categories_cache[kw] = name
        return self._categories_cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_fact(
        self,
        content: str,
        category: str = "general",
        tags: str = "",
    ) -> int:
        """Insert a fact and return its fact_id.

        Deduplicates by content (UNIQUE constraint). On duplicate, returns
        the existing fact_id without modifying the row. Extracts entities from
        the content and links them to the fact.
        """
        with self._lock:
            content = content.strip()
            if not content:
                raise ValueError("content must not be empty")

            # Auto-infer category from keywords if empty
            inferred_category = category
            if not category:
                inferred_category = self.extract_categories_from_content(content)

            try:
                cur = self._conn.execute(
                    """
                    INSERT INTO facts (content, category, tags, trust_score)
                    VALUES (?, ?, ?, ?)
                    """,
                    (content,
                     inferred_category,
                     tags,
                     self.default_trust),
                )
                self._conn.commit()
                fact_id: int = cur.lastrowid  # type: ignore[assignment]
            except sqlite3.IntegrityError:
                # Duplicate content — return existing id
                row = self._conn.execute(
                    "SELECT fact_id FROM facts WHERE content = ?", (content,)
                ).fetchone()
                return int(row["fact_id"])

            # Entity extraction and linking
            for name in self._extract_entities(content):
                entity_id = self._resolve_entity(name)
                self._link_fact_entity(fact_id, entity_id)

            # Compute HRR vector after entity linking
            self._compute_hrr_vector(fact_id, content)
            self._rebuild_bank(inferred_category)
            self.infer_and_add_relationships(fact_id)

            return fact_id

    def _search_facts_fts(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list[dict] | None:
        """Try FTS5 search; return None on failure (fts5 unavailable)."""
        try:
            params: list = [query, min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND f.category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT f.fact_id, f.content, f.category, f.tags,
                       f.trust_score, f.retrieval_count, f.helpful_count,
                       f.created_at, f.updated_at
                FROM facts f
                JOIN facts_fts fts ON fts.rowid = f.fact_id
                WHERE facts_fts MATCH ?
                  AND f.trust_score >= ?
                  {category_clause}
                ORDER BY fts.rank, f.trust_score DESC
                LIMIT ?
            """
            return self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return None

    def _search_facts_like(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list[sqlite3.Row]:
        """Fallback: LIKE search over facts.content + facts.tags (Python-side)."""
        # Split query into tokens for multi-keyword matching
        tokens = query.strip().split()
        like_conditions = " OR ".join(
            ["(f.content LIKE ? OR f.tags LIKE ?)"] * len(tokens)
        )
        params: list = []
        for tok in tokens:
            pat = f"%{tok}%"
            params.extend([pat, pat])

        category_clause = ""
        if category is not None:
            category_clause = "AND f.category = ?"
            params.append(category)

        params.extend([min_trust, limit])

        sql = f"""
            SELECT f.fact_id, f.content, f.category, f.tags,
                   f.trust_score, f.retrieval_count, f.helpful_count,
                   f.created_at, f.updated_at
            FROM facts f
            WHERE ({like_conditions})
              AND f.trust_score >= ?
              {category_clause}
            ORDER BY f.trust_score DESC, f.retrieval_count DESC
            LIMIT ?
        """
        return self._conn.execute(sql, params).fetchall()

    def search_facts(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Full-text search over facts (FTS5 with LIKE fallback).

        Returns a list of fact dicts ordered by relevance, then trust_score
        descending. Also increments retrieval_count for matched facts.
        """
        with self._lock:
            query = query.strip()
            if not query:
                return []

            # Try FTS5 first, fall back to LIKE
            # FIX (2026-05-13): _search_facts_fts returns [] (empty list) when no matches,
            # not None. Using `if rows is None` skipped the LIKE fallback for empty results.
            # Changed to `if not rows` to properly catch both None and empty list.
            rows: list[sqlite3.Row] | None = self._search_facts_fts(
                query, category, min_trust, limit
            )
            if not rows:
                # FTS returned empty list or unavailable — use LIKE fallback
                rows = self._search_facts_like(query, category, min_trust, limit)

            results = [self._row_to_dict(r) for r in rows]

            if results:
                ids = [r["fact_id"] for r in results]
                placeholders = ",".join("?" * len(ids))
                self._conn.execute(
                    f"UPDATE facts SET retrieval_count = retrieval_count + 1 WHERE fact_id IN ({placeholders})",
                    ids,
                )
                self._conn.commit()

            return results

    def update_fact(
        self,
        fact_id: int,
        content: str | None = None,
        trust_delta: float | None = None,
        tags: str | None = None,
        category: str | None = None,
    ) -> bool:
        """Partially update a fact. Trust is clamped to [0, 1].

        Returns True if the row existed, False otherwise.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False

            assignments: list[str] = ["updated_at = CURRENT_TIMESTAMP"]
            params: list = []

            if content is not None:
                assignments.append("content = ?")
                params.append(content.strip())
            if tags is not None:
                assignments.append("tags = ?")
                params.append(tags)
            if category is not None:
                assignments.append("category = ?")
                params.append(category)
            if trust_delta is not None:
                new_trust = _clamp_trust(row["trust_score"] + trust_delta)
                assignments.append("trust_score = ?")
                params.append(new_trust)

            params.append(fact_id)
            cur = self._conn.execute(
                f"UPDATE facts SET {', '.join(assignments)} WHERE fact_id = ?",
                params,
            )
            self._conn.commit()

            # Only rebuild bank if at least one row was actually changed
            if cur.rowcount == 0:
                return False

            # If content changed, re-extract entities
            if content is not None:
                self._conn.execute(
                    "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
                )
                for name in self._extract_entities(content):
                    entity_id = self._resolve_entity(name)
                    self._link_fact_entity(fact_id, entity_id)
                self._conn.commit()

            # Recompute HRR vector if content changed
            if content is not None:
                self._compute_hrr_vector(fact_id, content)
            # Rebuild bank for relevant category
            cat = category or self._conn.execute(
                "SELECT category FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()["category"]
            self._rebuild_bank(cat)

            return True

    def remove_fact(self, fact_id: int) -> bool:
        """Delete a fact and its entity links. Returns True if the row existed."""
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, category FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if row is None:
                return False

            self._conn.execute(
                "DELETE FROM fact_entities WHERE fact_id = ?", (fact_id,)
            )
            self._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
            self._conn.commit()
            self._rebuild_bank(row["category"])
            return True

    def list_facts(
        self,
        category: str | None = None,
        min_trust: float = 0.0,
        limit: int = 50,
    ) -> list[dict]:
        """Browse facts ordered by trust_score descending.

        Optionally filter by category and minimum trust score.
        """
        with self._lock:
            params: list = [min_trust]
            category_clause = ""
            if category is not None:
                category_clause = "AND category = ?"
                params.append(category)
            params.append(limit)

            sql = f"""
                SELECT fact_id, content, category, tags, trust_score,
                       retrieval_count, helpful_count, created_at, updated_at
                FROM facts
                WHERE trust_score >= ?
                  {category_clause}
                ORDER BY trust_score DESC
                LIMIT ?
            """
            rows = self._conn.execute(sql, params).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def record_feedback(self, fact_id: int, helpful: bool) -> dict:
        """Record user feedback and adjust trust asymmetrically.

        helpful=True  -> trust += 0.05, helpful_count += 1
        helpful=False -> trust -= 0.10

        Returns a dict with fact_id, old_trust, new_trust, helpful_count.
        Raises KeyError if fact_id does not exist.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT fact_id, trust_score, helpful_count FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"fact_id {fact_id} not found")

            old_trust: float = row["trust_score"]
            delta = _HELPFUL_DELTA if helpful else _UNHELPFUL_DELTA
            new_trust = _clamp_trust(old_trust + delta)

            helpful_increment = 1 if helpful else 0
            self._conn.execute(
                """
                UPDATE facts
                SET trust_score    = ?,
                    helpful_count  = helpful_count + ?,
                    updated_at     = CURRENT_TIMESTAMP
                WHERE fact_id = ?
                """,
                (new_trust, helpful_increment, fact_id),
            )
            self._conn.commit()

            return {
                "fact_id":      fact_id,
                "old_trust":    old_trust,
                "new_trust":    new_trust,
                "helpful_count": row["helpful_count"] + helpful_increment,
            }

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def _extract_entities(self, text: str) -> list[str]:
        """Extract entity candidates from text using optimized rules.

        Rules applied (in order):
        1. Capitalized multi-word phrases  e.g. "John Doe"
        2. Single capitalized words       e.g. "Python" (len >= 3, stop-word filtered)
        3. Chinese n-gram (2-4 char) window from jieba segmentation
        4. Double-quoted terms             e.g. "Python"
        5. Single-quoted terms             e.g. 'pytest'
        6. AKA patterns                    e.g. "Guido aka BDFL" -> two entities

        Chinese extraction uses n-gram sliding window over jieba segments to
        capture multi-word organization names (e.g. "华为技术有限公司"),
        then filters against _CHINESE_GENERIC_WORDS stop list.

        Returns a deduplicated list preserving first-seen order.
        """
        seen: set[str] = set()
        candidates: list[str] = []

        def _add(name: str) -> None:
            stripped = name.strip()
            if stripped and stripped.lower() not in seen:
                seen.add(stripped.lower())
                candidates.append(stripped)

        # 1. Multi-word capitalized phrases: "John Doe", "Anthropic Claude"
        for m in _RE_CAPITALIZED.finditer(text):
            _add(m.group(1))

        # 2. Single capitalized words (len >= 3), stop-word filtered: "Python", "Docker"
        for m in _RE_SINGLE_CAPITALIZED.finditer(text):
            word = m.group(1)
            if word.lower() not in _STOP_WORDS:
                _add(word)

        # 3. Chinese entity extraction: whole-word + n-gram hybrid
        #    - Jieba gives optimal word segmentation (e.g. "华为技术有限公司")
        #    - Extract whole segments of 2-4 chars directly (preference #1)
        #    - Also build n-gram windows for adjacent segments (preference #2)
        #    - Filter all results against _CHINESE_GENERIC_WORDS stop list
        chinese_stops = _CHINESE_GENERIC_WORDS

        # Collect positions of all CJK jieba segments
        chinese_segs: list[tuple[int, int, str]] = []  # (start, end, text)
        for word in jieba.cut(text):
            if re.match(r'[\u4e00-\u9fff]', word):
                start = text.find(word)
                if start != -1:
                    chinese_segs.append((start, start + len(word), word))

        seen_ngram: set[str] = set()

        def _add_ngram(ngram: str, score: int) -> None:
            """Add n-gram with deduplication, filtering generic words and sub-string matches.

            Sub-string check only applies to STOP WORDS OF 3+ CHARS.
            2-char stop words (e.g. "华为", "公司", "中国") are only exact-match filtered —
            they can appear as sub-strings of longer meaningful entities.
            """
            if ngram in seen_ngram:
                return
            # Exact match: reject immediately
            if ngram in chinese_stops:
                return
            # Sub-string match: only reject if n-gram contains a 3+-char stop word
            # (e.g. "美国公司" contains "公司" but "公司" is 2-char → allowed;
            #       "世界市场" contains "市场" but "市场" is 2-char → allowed)
            if any(len(stop) >= 3 and stop in ngram for stop in chinese_stops):
                return
            seen_ngram.add(ngram)
            _add(ngram)

        # Pass 1: whole jieba segments of 2-4 chars (highest confidence)
        for _pos, _end, word in chinese_segs:
            wlen = len(word)
            if 2 <= wlen <= 4 and word not in chinese_stops:
                _add_ngram(word, wlen * 2)  # double score = whole word bonus

        # Pass 2: n-gram windows of 2-3 chars built from adjacent segments
        # DISABLED — cross-segment n-grams produce too many noise combinations
        # (e.g. "华为由" from "华为|由" = meaningless).
        # The meaningful adjacent-segment cases are already captured by Pass 1
        # (jieba's own segmentation), so Pass 2 was redundant.
        # Kept as commented-out reference for future rule-based refinement.
        #
        # for n in (3, 2):
        #     for start_idx in range(len(chinese_segs)):
        #         pos = chinese_segs[start_idx][0]
        #         ngram_chars = 0
        #         seg_count = 0
        #         for end_idx in range(start_idx, len(chinese_segs)):
        #             s, e, w = chinese_segs[end_idx]
        #             expected_pos = chinese_segs[end_idx-1][1] if end_idx > start_idx else s
        #             if s != expected_pos:
        #                 break
        #             ngram_chars += len(w)
        #             seg_count += 1
        #             if ngram_chars == n:
        #                 ngram = text[pos:pos + n]
        #                 if re.match(r'^[\u4e00-\u9fff]{' + str(n) + r'}$', ngram):
        #                     _add_ngram(ngram, n)
        #             if ngram_chars > n:
        #                 break

        # Pass 3: extract 2-4 char PREFIXES from very long single segments
        # (e.g. jieba returns "华为技术有限公司" as one 8-char segment,
        # but we want "华为技术" as the meaningful prefix, not every substring)
        # Strategy: for a segment of len 5-10, extract prefix(2), prefix(3),
        # prefix(4) — in Chinese company names the identifying part is at the start.
        for _pos, _end, word in chinese_segs:
            wlen = len(word)
            if wlen >= 5 and wlen <= 12 and word not in chinese_stops:
                # Extract only prefix n-grams (first 2, 3, 4 chars)
                # Skip if any char pair matches a generic-word sub-string (heuristic)
                for substr_len in (4, 3, 2):
                    if wlen >= substr_len:
                        ngram = word[:substr_len]
                        if re.match(r'^[\u4e00-\u9fff]{' + str(substr_len) + r'}$', ngram):
                            _add_ngram(ngram, substr_len)


        # 4. Double-quoted terms: "Python"
        for m in _RE_DOUBLE_QUOTE.finditer(text):
            _add(m.group(1))

        # 5. Single-quoted terms: 'pytest'
        for m in _RE_SINGLE_QUOTE.finditer(text):
            _add(m.group(1))

        # 6. AKA patterns: "Guido aka BDFL" -> two entities
        for m in _RE_AKA.finditer(text):
            _add(m.group(1))
            _add(m.group(2))

        return candidates

    def _resolve_entity(self, name: str) -> int:
        """Find an existing entity by name or alias (case-insensitive) or create one.

        Returns the entity_id.
        """
        # Exact name match
        row = self._conn.execute(
            "SELECT entity_id FROM entities WHERE name LIKE ?", (name,)
        ).fetchone()
        if row is not None:
            return int(row["entity_id"])

        # Search aliases — aliases stored as comma-separated; use LIKE with % boundaries
        alias_row = self._conn.execute(
            """
            SELECT entity_id FROM entities
            WHERE ',' || aliases || ',' LIKE '%,' || ? || ',%'
            """,
            (name,),
        ).fetchone()
        if alias_row is not None:
            return int(alias_row["entity_id"])

        # Create new entity
        cur = self._conn.execute(
            "INSERT INTO entities (name) VALUES (?)", (name,)
        )
        self._conn.commit()
        return int(cur.lastrowid)  # type: ignore[return-value]

    def _link_fact_entity(self, fact_id: int, entity_id: int) -> None:
        """Insert into fact_entities, silently ignore if the link already exists."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO fact_entities (fact_id, entity_id)
            VALUES (?, ?)
            """,
            (fact_id, entity_id),
        )
        self._conn.commit()

    def _compute_hrr_vector(self, fact_id: int, content: str) -> None:
        """Compute and store HRR vector for a fact. No-op if numpy unavailable."""
        with self._lock:
            if not self._hrr_available:
                return

            # Get entities linked to this fact
            rows = self._conn.execute(
                """
                SELECT e.name FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE fe.fact_id = ?
                """,
                (fact_id,),
            ).fetchall()
            entities = [row["name"] for row in rows]

            vector = hrr.encode_fact(content, entities, self.hrr_dim)
            self._conn.execute(
                "UPDATE facts SET hrr_vector = ? WHERE fact_id = ?",
                (hrr.phases_to_bytes(vector), fact_id),
            )
            self._conn.commit()

    def _rebuild_bank(self, category: str) -> None:
        """Full rebuild of a category's memory bank from all its fact vectors."""
        with self._lock:
            if not self._hrr_available:
                return

            bank_name = f"cat:{category}"
            rows = self._conn.execute(
                "SELECT hrr_vector FROM facts WHERE category = ? AND hrr_vector IS NOT NULL",
                (category,),
            ).fetchall()

            if not rows:
                self._conn.execute("DELETE FROM memory_banks WHERE bank_name = ?", (bank_name,))
                self._conn.commit()
                return

            vectors = [hrr.bytes_to_phases(row["hrr_vector"]) for row in rows]

            # Filter out corrupted vectors with wrong dimensionality
            valid_vectors = []
            for vec in vectors:
                if vec.shape[0] != self.hrr_dim:
                    # Log but skip corrupted vector
                    print(f"WARNING: skipping corrupted vector dim={vec.shape[0]} expected={self.hrr_dim}")
                    continue
                valid_vectors.append(vec)
            if not valid_vectors:
                self._conn.execute("DELETE FROM memory_banks WHERE bank_name = ?", (bank_name,))
                self._conn.commit()
                return
            vectors = valid_vectors
            bank_vector = hrr.bundle(*vectors)
            fact_count = len(vectors)

            # Check SNR
            hrr.snr_estimate(self.hrr_dim, fact_count)

            self._conn.execute(
                """
                INSERT INTO memory_banks (bank_name, vector, dim, fact_count, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(bank_name) DO UPDATE SET
                    vector = excluded.vector,
                    dim = excluded.dim,
                    fact_count = excluded.fact_count,
                    updated_at = excluded.updated_at
                """,
                (bank_name, hrr.phases_to_bytes(bank_vector), self.hrr_dim, fact_count),
            )
            self._conn.commit()

    def rebuild_all_vectors(self, dim: int | None = None) -> int:
        """Recompute all HRR vectors + banks from text. For recovery/migration.

        Returns the number of facts processed.
        """
        with self._lock:
            if not self._hrr_available:
                return 0

            if dim is not None:
                self.hrr_dim = dim

            rows = self._conn.execute(
                "SELECT fact_id, content, category FROM facts"
            ).fetchall()

            categories: set[str] = set()
            for row in rows:
                self._compute_hrr_vector(row["fact_id"], row["content"])
                categories.add(row["category"])

            for category in categories:
                self._rebuild_bank(category)

            return len(rows)

    # ------------------------------------------------------------------
    # Dynamic Category Inference (from DB keywords)
    # ------------------------------------------------------------------

    def extract_categories_from_content(self, content: str) -> str:
        """Infer best category from content using DB keywords. Returns category name."""
        cats = self._load_categories_from_db()
        if not cats:
            return "general"
        content_lower = content.lower()
        best_cat: str | None = None
        best_score = 0
        cur_cat: str | None = None
        cur_score = 0
        i = 0
        while i < len(content_lower):
            # Check longest keyword first
            matched = False
            for kw in sorted(cats.keys(), key=len, reverse=True):
                if content_lower[i:i + len(kw)] == kw:
                    cat = cats[kw]
                    if cat != cur_cat:
                        if cur_cat and cur_score > best_score:
                            best_cat = cur_cat
                            best_score = cur_score
                        cur_cat = cat
                        cur_score = 0
                    cur_score += len(kw)
                    i += len(kw)
                    matched = True
                    break
            if not matched:
                i += 1
        if cur_cat and cur_score > best_score:
            best_cat = cur_cat
        return best_cat or "general"

    def infer_and_add_relationships(self, fact_id: int) -> int:
        """Extract entities from fact content, match relationship patterns,
        and INSERT into relationships table. Returns number of relationships added.

        Strategy:
        1. Try verb-based pattern matching first (X verb Y → relationship_type)
        2. Fall back to co_occurrence when >= 2 entities found but no verb match
        """
        row = self._conn.execute(
            "SELECT content FROM facts WHERE fact_id=?", (fact_id,)
        ).fetchone()
        if not row:
            return 0
        content = row[0]
        entities = self._extract_entities(content)
        added = 0

        # ── Strategy 1: verb-based pattern matching ──────────────────────
        # Pattern: "X verb Y" → infer relationship type from verb
        for verb, rel_type in self._ENTITY_VERB_MAP.items():
            # Build a pattern like:  (Entity) verb (Entity)
            # entities are already extracted; find them in the text
            verb_pos = content.find(verb)
            if verb_pos == -1:
                # Try with surrounding whitespace
                verb_pos = content.find(" " + verb + " ")
            if verb_pos == -1:
                continue
            # Find the closest two entities around the verb
            e1, e2 = self._find_entities_near_verb(content, entities, verb_pos)
            if e1 is not None and e2 is not None and e1 != e2:
                src = self._resolve_entity(e1)
                tgt = self._resolve_entity(e2)
                self._conn.execute("""
                    INSERT OR IGNORE INTO relationships
                        (source_entity, target_entity, relationship_type, source_fact_id)
                    VALUES (?, ?, ?, ?)
                    """, (src, tgt, rel_type, fact_id))
                added += 1
                break  # stop after first verb match

        # ── Strategy 2: co_occurrence fallback ───────────────────────────
        if added == 0 and len(entities) >= 2:
            # Pair all adjacent entity pairs with co_occurs
            for i in range(len(entities) - 1):
                e1, e2 = entities[i], entities[i + 1]
                if e1 == e2:
                    continue
                src = self._resolve_entity(e1)
                tgt = self._resolve_entity(e2)
                self._conn.execute("""
                    INSERT OR IGNORE INTO relationships
                        (source_entity, target_entity, relationship_type, source_fact_id)
                    VALUES (?, ?, ?, ?)
                    """, (src, tgt, "co_occurs", fact_id))
                added += 1
        return added

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    # Verb → relationship_type mapping (order matters: more specific first)
    _ENTITY_VERB_MAP: dict[str, str] = {
        # 主动创造 / 来源
        "创建":      "creates",
        "开发":      "develops",
        "发明":      "invents",
        "构建":      "builds",
        "建立":      "establishes",
        # 所属 / 成员
        "属于":      "member_of",
        "位于":      "located_in",
        "是":        "is_a",
        # 依赖 / 使用
        "用于":      "used_for",
        "基于":      "based_on",
        "依赖":      "depends_on",
        "采用":      "uses",
        # 关系人
        "出生于":    "born_in",
        "创立":      "founded_by",
        "资助":      "funds",
        "合作":      "collaborates_with",
        "竞争":      "competes_with",
        "收购":      "acquires",
        "集成":      "integrates_with",
        "替代":      "replaces",
        "兼容":      "compatible_with",
        "优于":      "better_than",
        "类似":      "similar_to",
        # 英文动词（降权，列在最后）
        "creates":   "creates",
        "develops":  "develops",
        "invents":   "invents",
        "builds":    "builds",
        "uses":      "uses",
        "based on":  "based_on",
        "depends on": "depends_on",
    }

    _re_inline = __import__("re").compile(
        r"(.+?)\s+(?:是|出生于|位于|创建|开发|属于|基于|用于)\s+(.+?)([：:]\s*)?$"
    )

    def _find_entities_near_verb(
        self, text: str, entities: list[str], verb_pos: int
    ) -> "tuple[str | None, str | None]":
        """Return the closest entity pair surrounding verb_pos in text."""
        if len(entities) < 2:
            return None, None
        # Find positions of each entity in the text
        positions: list[tuple[int, str]] = []
        for e in entities:
            start = 0
            while True:
                idx = text.find(e, start)
                if idx == -1:
                    break
                positions.append((idx, e))
                start = idx + 1
        if not positions:
            return None, None
        # Split by verb position
        left  = [(p, e) for p, e in positions if p < verb_pos]
        right = [(p, e) for p, e in positions if p > verb_pos]
        if not left or not right:
            return None, None
        # Closest on each side
        e1 = max(left, key=lambda x: x[0])[1]
        e2 = min(right, key=lambda x: x[0])[1]
        return e1, e2

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a sqlite3.Row to a plain dict."""
        return dict(row)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

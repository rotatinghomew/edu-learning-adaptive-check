"""SQLite 持久层。

事实表（responses、legacy_scores）只追加、不更新不删除；撤题通过独立的
withdrawals 标记表表达，历史成绩没有任何改写路径。数据库默认创建在
调用方指定的位置（演示程序使用临时目录），原始 CSV 保持不变。
"""
import json
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    skill_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    required_evidence INTEGER NOT NULL,
    mastery REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS items (
    item_id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL,
    modes TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS item_stats (
    item_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    correct INTEGER NOT NULL,
    PRIMARY KEY (item_id, mode)
);
CREATE TABLE IF NOT EXISTS withdrawals (
    item_id TEXT PRIMARY KEY,
    reason TEXT NOT NULL,
    kind TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL,
    family_id TEXT NOT NULL,
    learner_mode TEXT NOT NULL,
    target_skills TEXT NOT NULL,
    closed INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS responses (
    session_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    correct INTEGER NOT NULL,
    weight REAL NOT NULL,
    excluded INTEGER NOT NULL DEFAULT 0,
    exclusion_reason TEXT,
    PRIMARY KEY (session_id, item_id)
);
CREATE TABLE IF NOT EXISTS decision_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    code TEXT NOT NULL,
    explanation TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS legacy_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id TEXT NOT NULL,
    form_id TEXT NOT NULL,
    form_date TEXT NOT NULL,
    total_score INTEGER NOT NULL,
    passed INTEGER NOT NULL,
    skill_ids TEXT NOT NULL,
    scale_tag TEXT NOT NULL
);
"""


class Repository:
    def __init__(self, path):
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- 基础数据装载（重建型，仅作用于配置表） ------------------------
    def load_catalog(self, skills, items, stats_rows, legacy_scores):
        conn = self.connection
        conn.execute("DELETE FROM skills")
        conn.execute("DELETE FROM items")
        conn.execute("DELETE FROM item_stats")
        conn.executemany("INSERT INTO skills VALUES (?, ?, ?, ?)",
                         [(s["skill_id"], s["name"], s["required_evidence"],
                           s["mastery"]) for s in skills.values()])
        conn.executemany("INSERT INTO items VALUES (?, ?, ?)",
                         [(i["item_id"], i["skill_id"], "|".join(i["modes"]))
                          for i in items.values()])
        conn.executemany("INSERT INTO item_stats VALUES (?, ?, ?, ?)",
                         [(r["item_id"], r["mode"], r["attempts"],
                           r["correct"]) for r in stats_rows])
        # 历史成绩只追加：同一学员同一题单已存在则跳过，绝不覆盖。
        existing = {
            (row["learner_id"], row["form_id"])
            for row in conn.execute(
                "SELECT learner_id, form_id FROM legacy_scores")
        }
        conn.executemany(
            "INSERT INTO legacy_scores (learner_id, form_id, form_date, "
            "total_score, passed, skill_ids, scale_tag) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(s["learner_id"], s["form_id"], s["form_date"], s["total_score"],
              int(s["passed"]), "|".join(sorted(s["skill_ids"])),
              s["scale_tag"])
             for s in legacy_scores
             if (s["learner_id"], s["form_id"]) not in existing])
        conn.commit()

    # ---- 撤题 ----------------------------------------------------------
    def save_withdrawal(self, item_id, reason, kind):
        self.connection.execute(
            "INSERT OR REPLACE INTO withdrawals VALUES (?, ?, ?)",
            (item_id, reason, kind))
        self.connection.commit()

    def fetch_withdrawals(self):
        return {row["item_id"]: {"reason": row["reason"], "kind": row["kind"]}
                for row in self.connection.execute(
                    "SELECT item_id, reason, kind FROM withdrawals")}

    # ---- 会话与作答 ----------------------------------------------------
    def save_session(self, session):
        conn = self.connection
        conn.execute(
            "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
            (session.session_id, session.learner_id, session.family_id,
             session.learner_mode, json.dumps(list(session.target_skills)),
             int(session.closed)))
        conn.execute("DELETE FROM responses WHERE session_id = ?",
                     (session.session_id,))
        conn.execute("DELETE FROM decision_events WHERE session_id = ?",
                     (session.session_id,))
        conn.executemany(
            "INSERT INTO responses VALUES (?, ?, ?, ?, ?, ?)",
            [(session.session_id, item_id, int(r["correct"]), r["weight"],
              int(r["excluded"]), r["exclusion_reason"])
             for item_id, r in session.responses.items()])
        conn.executemany(
            "INSERT INTO decision_events (session_id, code, explanation) "
            "VALUES (?, ?, ?)",
            [(session.session_id, e["code"], e["explanation"])
             for e in session.events])
        conn.commit()

    def fetch_legacy_scores(self):
        rows = self.connection.execute(
            "SELECT learner_id, form_id, form_date, total_score, passed, "
            "skill_ids, scale_tag FROM legacy_scores ORDER BY id").fetchall()
        return [{
            "learner_id": r["learner_id"], "form_id": r["form_id"],
            "form_date": r["form_date"], "total_score": r["total_score"],
            "passed": bool(r["passed"]),
            "skill_ids": frozenset(r["skill_ids"].split("|")) if r["skill_ids"]
            else frozenset(),
            "scale_tag": r["scale_tag"],
        } for r in rows]

    def item_usage_counts(self):
        """跨会话曝光次数，供新会话做熟题轮换。"""
        rows = self.connection.execute(
            "SELECT item_id, COUNT(*) AS uses FROM responses "
            "WHERE excluded = 0 GROUP BY item_id").fetchall()
        return {r["item_id"]: r["uses"] for r in rows}

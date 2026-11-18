"""业务记录读取。

所有输入都是本地合成 CSV。读取层只负责字段与取值校验，
不做任何统计推断，保证原始事实与派生结果分开可核对。
"""
import csv
from pathlib import Path

# 每个业务表的必填字段；取值范围需要额外校验的字段在此声明。
_SCHEMAS = {
    "skills": {"required": {"skill_id", "name", "required_evidence", "mastery"}},
    "items": {"required": {"item_id", "skill_id", "supported_modes"}},
    "item_stats": {"required": {"item_id", "mode", "attempts", "correct"}},
    "courses": {"required": {"course_id", "name", "mentoring"}},
    "course_prereqs": {"required": {"course_id", "skill_id"}},
    "legacy_scores": {"required": {"learner_id", "form_id", "form_date",
                                   "total_score", "passed", "skill_ids",
                                   "scale_tag"}},
}

KNOWN_MODES = {"text", "large_print", "audio"}


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows


def _check_columns(table, rows):
    if not rows:
        raise ValueError(f"{table}: 空文件")
    required = _SCHEMAS[table]["required"]
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{table}: 缺少字段 {', '.join(sorted(missing))}")


def _as_int(table, row, field):
    try:
        return int(row[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{table}: 字段 {field} 必须是整数 (行: {row})") from exc


def _as_float(table, row, field):
    try:
        return float(row[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{table}: 字段 {field} 必须是数值 (行: {row})") from exc


def load_skills(path):
    table = "skills"
    rows = _read_csv(path)
    _check_columns(table, rows)
    skills = {}
    for row in rows:
        evidence = _as_int(table, row, "required_evidence")
        if evidence <= 0:
            raise ValueError(f"{table}: required_evidence 必须为正")
        mastery = _as_float(table, row, "mastery")
        if not 0.0 < mastery < 1.0:
            raise ValueError(f"{table}: mastery 必须在 0 与 1 之间")
        skills[row["skill_id"]] = {
            "skill_id": row["skill_id"],
            "name": row["name"],
            "required_evidence": evidence,
            "mastery": mastery,
        }
    return skills


def load_items(path, skills):
    table = "items"
    rows = _read_csv(path)
    _check_columns(table, rows)
    items = {}
    for row in rows:
        modes = tuple(m.strip() for m in row["supported_modes"].split("|") if m.strip())
        if not modes:
            raise ValueError(f"{table}: 题目 {row['item_id']} 未声明任何呈现方式")
        unknown = set(modes) - KNOWN_MODES
        if unknown:
            raise ValueError(f"{table}: 题目 {row['item_id']} 出现未知呈现方式 "
                             f"{', '.join(sorted(unknown))}")
        if row["skill_id"] not in skills:
            raise ValueError(f"{table}: 题目 {row['item_id']} 引用了未知技能 "
                             f"{row['skill_id']}")
        if len(set(modes)) != len(modes):
            raise ValueError(f"{table}: 题目 {row['item_id']} 呈现方式重复")
        items[row["item_id"]] = {
            "item_id": row["item_id"],
            "skill_id": row["skill_id"],
            "modes": modes,
        }
    return items


def load_item_stats(path, items):
    table = "item_stats"
    rows = _read_csv(path)
    _check_columns(table, rows)
    stats = {}
    for row in rows:
        attempts = _as_int(table, row, "attempts")
        correct = _as_int(table, row, "correct")
        if attempts < 0 or correct < 0 or correct > attempts:
            raise ValueError(f"{table}: 题目 {row['item_id']} 作答计数非法 "
                             f"(correct={correct}, attempts={attempts})")
        item = items.get(row["item_id"])
        if item is None:
            raise ValueError(f"{table}: 统计记录引用了未知题目 {row['item_id']}")
        if row["mode"] not in item["modes"]:
            raise ValueError(f"{table}: 题目 {row['item_id']} 不支持呈现方式 "
                             f"{row['mode']}")
        key = (row["item_id"], row["mode"])
        if key in stats:
            raise ValueError(f"{table}: {row['item_id']}/{row['mode']} 统计重复")
        stats[key] = {"item_id": row["item_id"], "mode": row["mode"],
                      "attempts": attempts, "correct": correct}
    return list(stats.values())


def load_courses(path):
    table = "courses"
    rows = _read_csv(path)
    _check_columns(table, rows)
    courses = {}
    for row in rows:
        mentoring = _as_int(table, row, "mentoring")
        if mentoring not in (0, 1):
            raise ValueError(f"{table}: mentoring 只能取 0/1")
        courses[row["course_id"]] = {
            "course_id": row["course_id"],
            "name": row["name"],
            "mentoring": bool(mentoring),
        }
    return courses


def load_course_prereqs(path, courses, skills):
    table = "course_prereqs"
    rows = _read_csv(path)
    _check_columns(table, rows)
    prereqs = {}
    for row in rows:
        if row["course_id"] not in courses:
            raise ValueError(f"{table}: 引用了未知课程 {row['course_id']}")
        if row["skill_id"] not in skills:
            raise ValueError(f"{table}: 引用了未知技能 {row['skill_id']}")
        prereqs.setdefault(row["course_id"], set()).add(row["skill_id"])
    for course_id, skill_ids in prereqs.items():
        prereqs[course_id] = frozenset(skill_ids)
    return prereqs


def load_legacy_scores(path, skills):
    """固定题单历史成绩。读取即冻结：没有任何更新路径会改写这些行。"""
    table = "legacy_scores"
    rows = _read_csv(path)
    _check_columns(table, rows)
    scores = []
    for row in rows:
        passed = _as_int(table, row, "passed")
        if passed not in (0, 1):
            raise ValueError(f"{table}: passed 只能取 0/1")
        total_score = _as_int(table, row, "total_score")
        if not 0 <= total_score <= 100:
            raise ValueError(f"{table}: total_score 必须在 0-100 之间")
        skill_ids = tuple(s.strip() for s in row["skill_ids"].split("|") if s.strip())
        unknown = set(skill_ids) - set(skills)
        if unknown:
            raise ValueError(f"{table}: 学员 {row['learner_id']} 的成绩引用未知技能 "
                             f"{', '.join(sorted(unknown))}")
        scores.append({
            "learner_id": row["learner_id"],
            "form_id": row["form_id"],
            "form_date": row["form_date"],
            "total_score": total_score,
            "passed": bool(passed),
            "skill_ids": frozenset(skill_ids),
            "scale_tag": row["scale_tag"],
        })
    return scores

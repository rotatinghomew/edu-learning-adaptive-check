"""选课建议：消费每个技能的证据，而不是一个总分布尔值。

建议分四类，工作人员可直接看到证据不足的具体技能：

* READY            先修资格有把握（全部必需技能合格）
* NEEDS_ASSESSMENT 存在证据不足的技能，先补充测查
* MENTORING        无证据不足、但有技能未掌握，仅适合带教课程
* NOT_RECOMMENDED  带教课程也不覆盖当前未掌握的技能
"""
from .session import QUALIFIED, NOT_MASTERED, INCONCLUSIVE

READY = "ready"
NEEDS_ASSESSMENT = "needs_assessment"
MENTORING = "mentoring_only"
NOT_RECOMMENDED = "not_recommended"

CATEGORY_LABELS = {
    READY: "有把握的先修资格",
    NEEDS_ASSESSMENT: "需要补充测查",
    MENTORING: "适合带教的课程选择",
    NOT_RECOMMENDED: "暂不建议",
}


def build_evidence_board(skills, board, legacy_scores=None, learner_id=None):
    """合并自适应会话判断与固定题单历史成绩。

    历史成绩保持当时含义、不可变：仅作为带出处的独立证据展示，
    不与当前尺度族的作答混算。
    """
    legacy_scores = legacy_scores or []
    legacy_by_skill = {}
    for score in legacy_scores:
        if score["learner_id"] != learner_id:
            continue
        for skill_id in score["skill_ids"]:
            legacy_by_skill.setdefault(skill_id, []).append(score)

    evidence = {}
    for skill_id, skill in skills.items():
        current = board.get(skill_id)
        history = legacy_by_skill.get(skill_id, [])
        history_note = [
            {"form_id": s["form_id"], "form_date": s["form_date"],
             "total_score": s["total_score"], "passed": s["passed"],
             "scale_tag": s["scale_tag"]}
            for s in history
        ]
        if current is not None:
            entry = dict(current)
            entry["provenance"] = "adaptive_session"
        elif history and all(s["passed"] for s in history):
            latest = sorted(history, key=lambda s: s["form_date"])[-1]
            entry = {
                "status": QUALIFIED,
                "reason": (f"固定题单 {latest['form_id']}（{latest['form_date']}）"
                           f"通过，总分 {latest['total_score']}；历史成绩按当时口径"
                           "保留，作为带出处的先修证据，未与自适应尺度混算"),
                "total_weight": None,
                "correct_weight": None,
                "wrong_weight": None,
                "p_above": None,
                "p_below": None,
                "provenance": "legacy_fixed_form",
            }
        elif history:
            latest = sorted(history, key=lambda s: s["form_date"])[-1]
            entry = {
                "status": INCONCLUSIVE,
                "reason": (f"固定题单 {latest['form_id']}（{latest['form_date']}）"
                           f"未通过（总分 {latest['total_score']}）；历史未通过不直接"
                           "判定未掌握，需在自适应测查中补充证据"),
                "total_weight": None,
                "correct_weight": None,
                "wrong_weight": None,
                "p_above": None,
                "p_below": None,
                "provenance": "legacy_fixed_form",
            }
        else:
            entry = {
                "status": INCONCLUSIVE,
                "reason": "该技能尚无任何测查记录或历史成绩，证据完全缺失",
                "total_weight": None,
                "correct_weight": None,
                "wrong_weight": None,
                "p_above": None,
                "p_below": None,
                "provenance": "none",
            }
        entry["legacy"] = history_note
        evidence[skill_id] = entry
    return evidence


def recommend_courses(courses, prereqs, evidence):
    """逐课程给出建议类别与可解释依据。"""
    recommendations = []
    for course_id in sorted(courses):
        needed = sorted(prereqs.get(course_id, frozenset()))
        qualified = [sid for sid in needed
                     if evidence[sid]["status"] == QUALIFIED]
        inconclusive = [sid for sid in needed
                        if evidence[sid]["status"] == INCONCLUSIVE]
        not_mastered = [sid for sid in needed
                        if evidence[sid]["status"] == NOT_MASTERED]

        if not needed:
            category = READY
            reason = "课程未声明先修技能"
        elif inconclusive:
            category = NEEDS_ASSESSMENT
            reason = ("先修技能证据不足，需补充测查: "
                      + "、".join(inconclusive))
        elif not_mastered:
            if courses[course_id]["mentoring"]:
                category = MENTORING
                reason = ("以下先修技能判定未掌握，仅可在带教支持下选择: "
                          + "、".join(not_mastered))
            else:
                category = NOT_RECOMMENDED
                reason = ("以下先修技能判定未掌握，且该课程不带教: "
                          + "、".join(not_mastered))
        else:
            category = READY
            reason = f"全部先修技能（{'、'.join(needed)}）均有合格证据"

        recommendations.append({
            "course_id": course_id,
            "course_name": courses[course_id]["name"],
            "mentoring": courses[course_id]["mentoring"],
            "category": category,
            "category_label": CATEGORY_LABELS[category],
            "required_skills": needed,
            "qualified_skills": qualified,
            "inconclusive_skills": inconclusive,
            "not_mastered_skills": not_mastered,
            "reason": reason,
        })
    return recommendations

"""端到端演示：从题库 CSV 到自适应测查与选课建议。

运行：python3 -m learning_adaptive_check.report
所有数据均为本地合成数据，SQLite 库创建在临时目录，原始 CSV 不被修改。
"""
import tempfile
from pathlib import Path

from .bank import QuestionBank, CONTENT_INVALID, NORMAL_RETIREMENT
from .calibration import calibrate_bank, MIN_ATTEMPTS
from .recommendation import build_evidence_board, recommend_courses, CATEGORY_LABELS
from .records import (load_skills, load_items, load_item_stats, load_courses,
                      load_course_prereqs, load_legacy_scores)
from .scales import decide_scale_families
from .session import AssessmentSession, handle_withdrawal, QUALIFIED, NOT_MASTERED, INCONCLUSIVE
from .storage import Repository

DATA = Path(__file__).parent.parent / "data"
STATUS_LABELS = {QUALIFIED: "合格", NOT_MASTERED: "未掌握",
                 INCONCLUSIVE: "证据不足"}
PROVENANCE_LABELS = {"adaptive_session": "自适应测查",
                     "legacy_fixed_form": "固定题单历史成绩",
                     "none": "无记录"}


def build_bank(data_dir):
    skills = load_skills(data_dir / "skills.csv")
    items = load_items(data_dir / "items.csv", skills)
    stats_rows = load_item_stats(data_dir / "item_stats.csv", items)
    calibration = calibrate_bank(stats_rows)
    scale_frozen = decide_scale_families(items, calibration)
    return QuestionBank(items, skills, calibration, scale_frozen), skills, stats_rows


def _print_calibration(bank):
    print("== 一、题库校准（小样本不判易/难） ==")
    under = [(key, cal) for key, cal in sorted(bank.calibration.items())
             if not cal["calibrated"]]
    for (item_id, mode), cal in under:
        raw = "无记录" if cal["raw_rate"] is None else f"答对率 {cal['raw_rate']:.0%}"
        print(f"  题 {item_id}/{mode}: {raw}，样本 {cal['attempts']} < "
              f"{MIN_ATTEMPTS}，难度判定={cal['difficulty']}，"
              f"证据权重仅 {cal['evidence_weight']:.2f}")
    easy = [(key, cal) for key, cal in sorted(bank.calibration.items())
            if cal["difficulty"] == "easy"]
    for (item_id, mode), cal in easy[:4]:
        print(f"  题 {item_id}/{mode}: 样本 {cal['attempts']}，"
              f"P(正确率>=80%)={cal['p_easy']:.1%}，判为容易题")
    print()
    print("== 二、尺度族与跨呈现方式适用范围 ==")
    for family in bank.family_summary():
        print(f"  尺度族 {family['family_id']}: {family['scope']}")
    for entry in bank.scales.bridge_log:
        if not entry["bridged"]:
            print(f"  桥接未通过 题 {entry['item_id']}: {entry['reason']}")
    print()


def _print_board(board, skills):
    for skill_id in sorted(skills):
        entry = board[skill_id]
        print(f"  技能 {skill_id}（{skills[skill_id]['name']}）: "
              f"{STATUS_LABELS[entry['status']]} "
              f"[{PROVENANCE_LABELS[entry['provenance']]}] — {entry['reason']}")
        for legacy in entry["legacy"]:
            print(f"      历史出处: {legacy['form_id']} {legacy['form_date']} "
                  f"总分 {legacy['total_score']} "
                  f"{'通过' if legacy['passed'] else '未通过'}（{legacy['scale_tag']}）")


def _print_recommendations(recommendations):
    order = ["ready", "needs_assessment", "mentoring_only", "not_recommended"]
    grouped = {}
    for rec in recommendations:
        grouped.setdefault(rec["category"], []).append(rec)
    for category in order:
        for rec in grouped.get(category, []):
            print(f"  [{CATEGORY_LABELS[category]}] {rec['course_id']} "
                  f"{rec['course_name']}: {rec['reason']}")


def _drive(session, bank, answer_plan, stop_skill=None, stop_after=0):
    """按答案脚本驱动会话；可在某技能累计 stop_after 题后暂停（用于撤题演示）。"""
    answered = {}
    paused = False
    while not paused:
        item_id, why = session.next_item()
        if item_id is None:
            print(f"  停止出题: {why}")
            break
        skill_id = bank.items[item_id]["skill_id"]
        idx = answered.get(skill_id, 0)
        default = idx % 2 == 0
        correct = answer_plan.get(skill_id, [default] * 99)[
            idx] if idx < len(answer_plan.get(skill_id, [])) else default
        accepted, code, explanation = session.submit(item_id, correct)
        if not accepted:
            print(f"  提交被拒绝（{code}）: {explanation}")
            break
        answered[skill_id] = idx + 1
        print(f"  出题 {item_id}（技能 {skill_id}）: {why}")
        print(f"    -> 作答{'正确' if correct else '错误'}（{explanation}）")
        again, again_code, again_explain = session.submit(item_id, not correct)
        assert not again and again_code == "DUPLICATE_REJECTED"
        print(f"    -> 重复提交被拦截: {again_explain}")
        if stop_skill and skill_id == stop_skill and answered[skill_id] >= stop_after:
            paused = True
    return answered


def run_adaptive_learner(repo, bank, skills, courses, prereqs, legacy):
    print("== 三、自适应测查（学员 L001，音频呈现，音频独立尺度族） ==")
    family_id = bank.scales.family_for_mode("audio")
    session = AssessmentSession(
        "S-L001-01", "L001", bank, family_id, "audio",
        target_skills=["S1", "S2", "S3"],
        usage_counts=repo.item_usage_counts())

    # 先答 S3 一题，随后该题被确认内容失效，演示进行中撤题的证据剔除。
    plan_phase1 = {"S3": [False]}
    _drive(session, bank, plan_phase1, stop_skill="S3", stop_after=1)
    withdrawn_item = next(iter(
        item for item, r in session.responses.items()
        if bank.items[item]["skill_id"] == "S3"))
    print()
    print(f"  -- 测查进行期间撤题：题 {withdrawn_item} 被确认内容失效 --")
    result = handle_withdrawal(session, withdrawn_item,
                               "支付界面改版，题目内容已失效", CONTENT_INVALID)
    print(f"     受影响技能: {'、'.join(result['affected_skills'])}；"
          f"剔除证据: {'、'.join(result['excluded']) or '无'}")
    for event in session.events[-2:]:
        print(f"     [{event['code']}] {event['explanation']}")

    # 正常下架：本会话未使用的音频题，后续选题自动跳过，已作答证据不受影响。
    other_audio = next(item for item in bank.eligible_items(family_id, "audio")
                       if item not in session.responses)
    handle_withdrawal(session, other_audio, "音频版权到期，正常下架",
                      NORMAL_RETIREMENT)
    print(f"  -- 题 {other_audio} 正常下架：撤题前无作答，无证据需要处理 --")
    print()

    # 撤题后继续：S1 全对合格、S2 错三题判未掌握、S3/S4 题尽则证据不足。
    plan_phase2 = {
        "S1": [True, True, True, True],
        "S2": [False, False, False],
        "S3": [False, True],
    }
    _drive(session, bank, plan_phase2)
    close_code, close_explain = session.close()
    print(f"  结束测查 [{close_code}]: {close_explain}")
    repo.save_withdrawal(withdrawn_item, "支付界面改版，题目内容已失效",
                         CONTENT_INVALID)
    repo.save_withdrawal(other_audio, "音频版权到期，正常下架", NORMAL_RETIREMENT)
    repo.save_session(session)

    print()
    print("== 四、L001 技能证据板与选课建议 ==")
    print("  注：S4 未纳入本次测查目标且无历史成绩，作为证据完全缺失示例。")
    evidence = build_evidence_board(skills, session.skill_board(), legacy, "L001")
    _print_board(evidence, skills)
    print("  -- 课程建议 --")
    _print_recommendations(recommend_courses(courses, prereqs, evidence))
    print()


def run_legacy_learner(repo, bank, skills, courses, prereqs, legacy):
    print("== 五、固定题单历史成绩学员（L003，历史成绩不可变） ==")
    family_id = bank.scales.family_for_mode("text")
    # I08 已在 L001 流程中撤下，新会话不能再选它，验证撤题对新会话生效。
    session = AssessmentSession(
        "S-L003-01", "L003", bank, family_id, "text",
        target_skills=["S3", "S4"], usage_counts=repo.item_usage_counts())
    print(f"  新开自适应会话只补历史未覆盖技能 S3/S4；绑定尺度族 {family_id}，"
          "不与历史题单尺度混算；已撤下的题不会出现。")
    plan = {"S3": [False, False, False], "S4": [True, True, True]}
    _drive(session, bank, plan)
    close_code, close_explain = session.close()
    print(f"  结束测查 [{close_code}]: {close_explain}")
    repo.save_session(session)

    evidence = build_evidence_board(skills, session.skill_board(), legacy, "L003")
    _print_board(evidence, skills)
    print("  -- 课程建议 --")
    _print_recommendations(recommend_courses(courses, prereqs, evidence))
    print()


def main():
    data_dir = DATA
    bank, skills, stats_rows = build_bank(data_dir)
    courses = load_courses(data_dir / "courses.csv")
    prereqs = load_course_prereqs(data_dir / "course_prereqs.csv", courses, skills)
    legacy = load_legacy_scores(data_dir / "legacy_scores.csv", skills)
    with tempfile.TemporaryDirectory() as directory:
        repo = Repository(Path(directory) / "adaptive.db")
        repo.load_catalog(skills, bank.items, stats_rows, legacy)
        _print_calibration(bank)
        run_adaptive_learner(repo, bank, skills, courses, prereqs,
                             repo.fetch_legacy_scores())
        # 撤题标记从持久层恢复到题库，确保后续会话不能再用已撤下的题。
        for item_id, entry in repo.fetch_withdrawals().items():
            if not bank.is_available(item_id):
                continue
            bank.withdraw(item_id, entry["reason"], entry["kind"])
        run_legacy_learner(repo, bank, skills, courses, prereqs,
                           repo.fetch_legacy_scores())
        repo.close()
    print("演示结束：原始 CSV 未修改，临时 SQLite 库已随目录清理。")


if __name__ == "__main__":
    main()

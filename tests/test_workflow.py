"""端到端工作流测试：题库统计 -> 自适应测查 -> 撤题 -> 选课建议。"""
import unittest
from pathlib import Path

from learning_adaptive_check.bank import CONTENT_INVALID, NORMAL_RETIREMENT
from learning_adaptive_check.recommendation import (
    build_evidence_board, recommend_courses,
    READY, NEEDS_ASSESSMENT, MENTORING, NOT_RECOMMENDED)
from learning_adaptive_check.records import (
    load_skills, load_items, load_item_stats, load_courses,
    load_course_prereqs, load_legacy_scores)
from learning_adaptive_check.calibration import calibrate_bank
from learning_adaptive_check.scales import decide_scale_families
from learning_adaptive_check.bank import QuestionBank
from learning_adaptive_check.session import (
    AssessmentSession, handle_withdrawal,
    QUALIFIED, NOT_MASTERED, INCONCLUSIVE)
from learning_adaptive_check import report

DATA = Path(__file__).parent.parent / "data"


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skills = load_skills(DATA / "skills.csv")
        cls.items = load_items(DATA / "items.csv", cls.skills)
        cls.stats = load_item_stats(DATA / "item_stats.csv", cls.items)
        cls.courses = load_courses(DATA / "courses.csv")
        cls.prereqs = load_course_prereqs(
            DATA / "course_prereqs.csv", cls.courses, cls.skills)
        cls.legacy = load_legacy_scores(DATA / "legacy_scores.csv", cls.skills)

    def _bank(self):
        cal = calibrate_bank(self.stats)
        return QuestionBank(self.items, self.skills, cal,
                            decide_scale_families(self.items, cal))

    def test_full_adaptive_flow_with_mid_session_withdrawal(self):
        bank = self._bank()
        family = bank.scales.family_for_mode("audio")
        session = AssessmentSession(
            "E2E1", "L001", bank, family, "audio",
            target_skills=["S1", "S2", "S3"])

        # 1) 覆盖优先：前三题覆盖三个技能。
        first_skills = set()
        for _ in range(3):
            item_id, _ = session.next_item()
            session.submit(item_id, True)
            first_skills.add(bank.items[item_id]["skill_id"])
        self.assertEqual(first_skills, {"S1", "S2", "S3"})

        # 2) 进行中撤下一道已作答题（内容失效），证据被剔除并重算。
        target = next(item for item, r in session.responses.items()
                      if bank.items[item]["skill_id"] == "S3")
        result = handle_withdrawal(session, target, "内容失效", CONTENT_INVALID)
        self.assertEqual(result["affected_skills"], ["S3"])
        self.assertIn(target, result["excluded"])

        # 3) 继续作答到题尽：任何耗尽技能都只能是证据不足，不会自动通过。
        # S3 再撤下一道可用题，使剩余音频题不足以凑满证据量。
        remaining_s3 = [item for item in bank.eligible_items(family, "audio", "S3")
                        if item not in session.responses]
        handle_withdrawal(session, remaining_s3[0], "题面过时", NORMAL_RETIREMENT)
        # 技能级默认作答：S1 全对（合格）、S2 全错（未掌握）、S3 对错交替。
        default_answer = {"S1": True, "S2": False, "S3": True}
        answered = {}
        while True:
            item_id, why = session.next_item()
            if item_id is None:
                self.assertIn("不会自动判为通过", why)
                break
            skill = bank.items[item_id]["skill_id"]
            idx = answered.get(skill, 0)
            correct = (False if skill == "S3" and idx % 2
                       else default_answer[skill])
            accepted, code, _ = session.submit(item_id, correct)
            self.assertTrue(accepted, msg=f"unexpected {code}")
            answered[skill] = idx + 1
        close_code, _ = session.close()
        self.assertEqual(close_code, "CLOSED_INCONCLUSIVE")

        board = session.skill_board()
        self.assertEqual(board["S1"]["status"], QUALIFIED)
        self.assertEqual(board["S2"]["status"], NOT_MASTERED)
        self.assertEqual(board["S3"]["status"], INCONCLUSIVE)

        evidence = build_evidence_board(self.skills, board, self.legacy, "L001")
        recs = {r["course_id"]: r["category"]
                for r in recommend_courses(self.courses, self.prereqs, evidence)}
        self.assertEqual(recs["C01"], READY)
        self.assertEqual(recs["C03"], MENTORING)
        self.assertEqual(recs["C04"], NEEDS_ASSESSMENT)

    def test_legacy_learner_combines_history_and_new_evidence_without_mixing(self):
        bank = self._bank()
        bank.withdraw("I08", "内容失效", CONTENT_INVALID)
        family = bank.scales.family_for_mode("text")
        session = AssessmentSession(
            "E2E2", "L003", bank, family, "text", target_skills=["S3", "S4"])
        plan = {"S3": [False, False, False], "S4": [True, True, True]}
        answered = {}
        while True:
            item_id, _ = session.next_item()
            if item_id is None:
                break
            self.assertNotEqual(item_id, "I08")
            skill = bank.items[item_id]["skill_id"]
            session.submit(item_id, plan[skill][answered.get(skill, 0)])
            answered[skill] = answered.get(skill, 0) + 1
        session.close()
        evidence = build_evidence_board(
            self.skills, session.skill_board(), self.legacy, "L003")
        self.assertEqual(evidence["S1"]["provenance"], "legacy_fixed_form")
        self.assertEqual(evidence["S3"]["status"], NOT_MASTERED)
        self.assertEqual(evidence["S4"]["status"], QUALIFIED)
        recs = {r["course_id"]: r["category"]
                for r in recommend_courses(self.courses, self.prereqs, evidence)}
        self.assertEqual(recs["C01"], READY)
        self.assertEqual(recs["C02"], MENTORING)
        self.assertEqual(recs["C04"], NOT_RECOMMENDED)

    def test_report_module_runs_end_to_end(self):
        # 报告主流程无异常即覆盖演示脚本的完整链路。
        report.main()


if __name__ == "__main__":
    unittest.main()

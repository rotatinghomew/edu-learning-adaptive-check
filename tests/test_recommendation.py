import unittest
from pathlib import Path

from learning_adaptive_check.recommendation import (
    build_evidence_board, recommend_courses,
    READY, NEEDS_ASSESSMENT, MENTORING, NOT_RECOMMENDED)
from learning_adaptive_check.records import (
    load_skills, load_courses, load_course_prereqs, load_legacy_scores)
from learning_adaptive_check.session import QUALIFIED, NOT_MASTERED, INCONCLUSIVE

DATA = Path(__file__).parent.parent / "data"


class RecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skills = load_skills(DATA / "skills.csv")
        cls.courses = load_courses(DATA / "courses.csv")
        cls.prereqs = load_course_prereqs(
            DATA / "course_prereqs.csv", cls.courses, cls.skills)
        cls.legacy = load_legacy_scores(DATA / "legacy_scores.csv", cls.skills)

    def _board(self, mapping):
        return {
            sid: {"status": status, "reason": "test", "total_weight": 3.0,
                  "correct_weight": 2.0, "wrong_weight": 1.0,
                  "p_above": 0.9, "p_below": 0.1}
            for sid, status in mapping.items()}

    def test_four_recommendation_categories(self):
        # C01(S1) 全合格 -> ready；C02(S1,S3,S4) 含证据不足 -> 补充测查；
        # C03(S2) 未掌握但带教 -> mentoring；C04(S3) 未掌握且不带教 -> 不建议。
        board = self._board({
            "S1": QUALIFIED,
            "S2": NOT_MASTERED,
            "S3": NOT_MASTERED,
            "S4": INCONCLUSIVE,
        })
        evidence = build_evidence_board(self.skills, board)
        recs = {r["course_id"]: r for r in
                recommend_courses(self.courses, self.prereqs, evidence)}
        self.assertEqual(recs["C01"]["category"], READY)
        self.assertEqual(recs["C02"]["category"], NEEDS_ASSESSMENT)
        self.assertEqual(recs["C02"]["inconclusive_skills"], ["S4"])
        self.assertIn("S4", recs["C02"]["reason"])
        self.assertEqual(recs["C03"]["category"], MENTORING)
        self.assertEqual(recs["C04"]["category"], NOT_RECOMMENDED)

    def test_inconclusive_lists_exact_skill_locations(self):
        board = self._board({
            "S1": QUALIFIED, "S2": QUALIFIED,
            "S3": INCONCLUSIVE, "S4": INCONCLUSIVE})
        evidence = build_evidence_board(self.skills, board)
        recs = {r["course_id"]: r for r in
                recommend_courses(self.courses, self.prereqs, evidence)}
        self.assertEqual(recs["C02"]["inconclusive_skills"], ["S3", "S4"])
        self.assertEqual(recs["C04"]["inconclusive_skills"], ["S3"])

    def test_legacy_pass_is_kept_separate_and_never_overwritten(self):
        evidence = build_evidence_board(
            self.skills, self._board({"S3": NOT_MASTERED, "S4": QUALIFIED}),
            self.legacy, "L003")
        s1 = evidence["S1"]
        self.assertEqual(s1["status"], QUALIFIED)
        self.assertEqual(s1["provenance"], "legacy_fixed_form")
        self.assertEqual(s1["legacy"][0]["form_id"], "F2025Q4")
        # 历史成绩只作展示出处，不与自适应证据混算。
        self.assertIsNone(s1["p_above"])
        s3 = evidence["S3"]
        self.assertEqual(s3["provenance"], "adaptive_session")

    def test_legacy_failure_does_not_directly_fail_skill(self):
        evidence = build_evidence_board(self.skills, {}, self.legacy, "L005")
        self.assertEqual(evidence["S1"]["status"], INCONCLUSIVE)
        self.assertEqual(evidence["S1"]["provenance"], "legacy_fixed_form")
        self.assertIn("未通过", evidence["S1"]["reason"])

    def test_no_records_means_explicit_missing_evidence(self):
        evidence = build_evidence_board(self.skills, {}, [], "L999")
        self.assertEqual(evidence["S4"]["status"], INCONCLUSIVE)
        self.assertEqual(evidence["S4"]["provenance"], "none")


if __name__ == "__main__":
    unittest.main()

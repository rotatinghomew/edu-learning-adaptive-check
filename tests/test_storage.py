import tempfile
import unittest
from pathlib import Path

from learning_adaptive_check.bank import QuestionBank, CONTENT_INVALID
from learning_adaptive_check.calibration import calibrate_bank
from learning_adaptive_check.records import (
    load_skills, load_items, load_item_stats, load_courses,
    load_course_prereqs, load_legacy_scores)
from learning_adaptive_check.scales import decide_scale_families
from learning_adaptive_check.session import AssessmentSession, handle_withdrawal
from learning_adaptive_check.storage import Repository

DATA = Path(__file__).parent.parent / "data"


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.skills = load_skills(DATA / "skills.csv")
        self.items = load_items(DATA / "items.csv", self.skills)
        self.stats = load_item_stats(DATA / "item_stats.csv", self.items)
        self.legacy = load_legacy_scores(DATA / "legacy_scores.csv", self.skills)

    def _bank(self):
        cal = calibrate_bank(self.stats)
        return QuestionBank(self.items, self.skills, cal,
                            decide_scale_families(self.items, cal))

    def test_legacy_scores_append_only_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Repository(Path(tmp) / "t.db")
            repo.load_catalog(self.skills, self.items, self.stats, self.legacy)
            first = repo.fetch_legacy_scores()
            # 再次装载同一份历史：跳过而不是覆盖，行数不变、值不变。
            repo.load_catalog(self.skills, self.items, self.stats, self.legacy)
            second = repo.fetch_legacy_scores()
            self.assertEqual(len(first), len(second))
            self.assertEqual(
                [(r["learner_id"], r["total_score"], r["passed"])
                 for r in first],
                [(r["learner_id"], r["total_score"], r["passed"])
                 for r in second])
            repo.close()

    def test_session_round_trip_preserves_excluded_evidence_and_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Repository(Path(tmp) / "t.db")
            repo.load_catalog(self.skills, self.items, self.stats, self.legacy)
            bank = self._bank()
            family = bank.scales.family_for_mode("audio")
            session = AssessmentSession(
                "RT1", "L001", bank, family, "audio",
                target_skills=["S3"], usage_counts=repo.item_usage_counts())
            item_id, _ = session.next_item()
            session.submit(item_id, True)
            handle_withdrawal(session, item_id, "内容失效", CONTENT_INVALID)
            repo.save_withdrawal(item_id, "内容失效", CONTENT_INVALID)
            repo.save_session(session)

            rows = repo.connection.execute(
                "SELECT excluded, exclusion_reason FROM responses"
            ).fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["excluded"], 1)
            self.assertIsNotNone(rows[0]["exclusion_reason"])
            events = repo.connection.execute(
                "SELECT code FROM decision_events").fetchall()
            self.assertIn("WITHDRAWAL_EVIDENCE_REMOVED",
                          [r["code"] for r in events])
            self.assertEqual(repo.fetch_withdrawals()[item_id]["kind"],
                             CONTENT_INVALID)
            self.assertGreaterEqual(repo.item_usage_counts().get(item_id, 0), 0)
            repo.close()

    def test_usage_counts_feed_future_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Repository(Path(tmp) / "t.db")
            repo.load_catalog(self.skills, self.items, self.stats, self.legacy)
            bank = self._bank()
            family = bank.scales.family_for_mode("text")
            session = AssessmentSession(
                "RT2", "L009", bank, family, "text", target_skills=["S4"])
            item_id, _ = session.next_item()
            session.submit(item_id, True)
            repo.save_session(session)
            counts = repo.item_usage_counts()
            self.assertEqual(counts[item_id], 1)
            repo.close()


if __name__ == "__main__":
    unittest.main()

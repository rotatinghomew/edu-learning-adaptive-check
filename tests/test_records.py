import csv
import tempfile
import unittest
from pathlib import Path

from learning_adaptive_check.records import (
    load_skills, load_items, load_item_stats, load_legacy_scores,
    load_courses, load_course_prereqs)

DATA = Path(__file__).parent.parent / "data"


def _write(tmp, name, content):
    path = Path(tmp) / name
    path.write_text(content, encoding="utf-8")
    return path


class RecordsValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skills = load_skills(DATA / "skills.csv")
        cls.items = load_items(DATA / "items.csv", cls.skills)

    def test_missing_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "bad.csv", "skill_id,name\nS1,x\n")
            with self.assertRaisesRegex(ValueError, "缺少字段"):
                load_skills(path)

    def test_empty_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "empty.csv", "")
            with self.assertRaisesRegex(ValueError, "空文件"):
                load_skills(path)

    def test_illegal_counts_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(
                tmp, "stats.csv",
                "item_id,mode,attempts,correct\nI01,text,5,9\n")
            with self.assertRaisesRegex(ValueError, "计数非法"):
                load_item_stats(path, self.items)

    def test_unknown_mode_and_unsupported_mode_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(
                tmp, "items.csv",
                "item_id,skill_id,supported_modes\nIX1,S1,video\n")
            with self.assertRaisesRegex(ValueError, "未知呈现方式"):
                load_items(path, self.skills)
            path2 = _write(
                tmp, "stats2.csv",
                "item_id,mode,attempts,correct\nI01,audio,10,5\n")
            with self.assertRaisesRegex(ValueError, "不支持呈现方式"):
                load_item_stats(path2, self.items)

    def test_legacy_scores_validated_and_frozen_shape(self):
        scores = load_legacy_scores(DATA / "legacy_scores.csv", self.skills)
        self.assertEqual(len(scores), 2)
        self.assertIsInstance(scores[0]["skill_ids"], frozenset)
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(
                tmp, "legacy.csv",
                "learner_id,form_id,form_date,total_score,passed,skill_ids,"
                "scale_tag\nL1,F1,2025-01-01,120,1,S1,t\n")
            with self.assertRaisesRegex(ValueError, "total_score"):
                load_legacy_scores(path, self.skills)

    def test_course_referential_integrity(self):
        courses = load_courses(DATA / "courses.csv")
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "prereq.csv", "course_id,skill_id\nC99,S1\n")
            with self.assertRaisesRegex(ValueError, "未知课程"):
                load_course_prereqs(path, courses, self.skills)

    def test_synthetic_data_loads_cleanly(self):
        stats = load_item_stats(DATA / "item_stats.csv", self.items)
        self.assertGreater(len(stats), 30)
        # 每条统计的题目都存在，且模式被题目支持。
        for row in stats:
            self.assertIn(row["mode"], self.items[row["item_id"]]["modes"])
        # CSV 仍是原始输入，没有被任何读取操作改写（行数稳定）。
        with (DATA / "items.csv").open(encoding="utf-8") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))),
                             len(self.items))


if __name__ == "__main__":
    unittest.main()

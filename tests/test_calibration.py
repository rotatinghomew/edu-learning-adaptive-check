import unittest

from learning_adaptive_check.calibration import (
    calibrate_mode, evidence_weight, beta_posterior,
    betainc_regularized, credible_interval, MIN_ATTEMPTS,
    EASY_THRESHOLD, HARD_THRESHOLD)
from learning_adaptive_check.scales import decide_scale_families, BRIDGE_MAX_DIFF
from learning_adaptive_check.records import load_skills, load_items, load_item_stats
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"


class CalibrationTests(unittest.TestCase):
    def test_beta_cdf_matches_known_integer_values(self):
        # I_x(a, b) 在整数参数下有闭式结果，直接核对数值实现。
        self.assertAlmostEqual(betainc_regularized(2, 2, 0.5), 0.5, places=6)
        self.assertAlmostEqual(betainc_regularized(3, 1, 0.7), 0.7 ** 3, places=6)
        self.assertAlmostEqual(betainc_regularized(1, 4, 0.2),
                               1 - 0.8 ** 4, places=6)
        self.assertEqual(betainc_regularized(2, 3, 0.0), 0.0)
        self.assertEqual(betainc_regularized(2, 3, 1.0), 1.0)

    def test_credible_interval_brackets_mean_and_grows_without_data(self):
        narrow = credible_interval(*beta_posterior(100, 80))
        wide = credible_interval(*beta_posterior(0, 0))
        self.assertLess(narrow[0], narrow[1])
        self.assertGreater(wide[1] - wide[0], narrow[1] - narrow[0])
        posterior = beta_posterior(100, 80)
        mean_point = posterior[0] / (posterior[0] + posterior[1])
        self.assertLess(narrow[0], mean_point)
        self.assertLess(mean_point, narrow[1])

    def test_small_sample_high_rate_is_not_easy(self):
        # 3/3 全对但样本量不足：不能因偶然高答对率判成容易题。
        cal = calibrate_mode(3, 3)
        self.assertFalse(cal["calibrated"])
        self.assertEqual(cal["difficulty"], "undetermined")
        self.assertLess(cal["p_easy"], 0.85)
        self.assertLess(cal["evidence_weight"], 1.0)

    def test_calibrated_easy_and_hard_need_one_sided_posterior(self):
        easy = calibrate_mode(100, 95)
        hard = calibrate_mode(100, 10)
        self.assertTrue(easy["calibrated"] and hard["calibrated"])
        self.assertEqual(easy["difficulty"], "easy")
        self.assertEqual(hard["difficulty"], "hard")
        # 恰好 80% 答对率、样本有限时，单侧概率不足以越线，仍判 undetermined。
        borderline = calibrate_mode(MIN_ATTEMPTS, int(MIN_ATTEMPTS * EASY_THRESHOLD))
        self.assertEqual(borderline["difficulty"], "undetermined")
        # 低答对率且单侧后验越过难判据时判难；恰在 40% 附近则保持未定。
        near_hard = calibrate_mode(40, 8)  # 20% 答对率
        self.assertEqual(near_hard["difficulty"], "hard")
        self.assertGreaterEqual(near_hard["p_hard"], 0.85)
        borderline_hard = calibrate_mode(40, 16)  # 40% 答对率，证据不偏向
        self.assertEqual(borderline_hard["difficulty"], "undetermined")

    def test_evidence_weight_ramp(self):
        self.assertEqual(evidence_weight(0), 0.5)
        self.assertLess(evidence_weight(10), evidence_weight(20))
        self.assertEqual(evidence_weight(MIN_ATTEMPTS), 1.0)
        self.assertEqual(evidence_weight(999), 1.0)

    def test_thresholds_are_inside_unit_interval(self):
        self.assertGreater(HARD_THRESHOLD, 0.0)
        self.assertLess(EASY_THRESHOLD, 1.0)
        self.assertLess(HARD_THRESHOLD, EASY_THRESHOLD)


class ScaleFamilyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skills = load_skills(DATA / "skills.csv")
        cls.items = load_items(DATA / "items.csv", cls.skills)
        cls.stats = load_item_stats(DATA / "item_stats.csv", cls.items)
        cls.cal = {(r["item_id"], r["mode"]):
                   calibrate_mode(r["attempts"], r["correct"]) for r in cls.stats}
        cls.frozen = decide_scale_families(cls.items, cls.cal)

    def test_text_and_large_print_pool_with_bridge_evidence(self):
        family_text = self.frozen.family_for_mode("text")
        family_lp = self.frozen.family_for_mode("large_print")
        self.assertEqual(family_text, family_lp)
        family = self.frozen.families[family_text]
        self.assertGreaterEqual(len(family["bridge_items"]), 2)
        self.assertIn("桥接题", family["scope"])

    def test_audio_is_separate_when_bridge_fails(self):
        family_audio = self.frozen.family_for_mode("audio")
        family_text = self.frozen.family_for_mode("text")
        self.assertNotEqual(family_audio, family_text)
        failed = {e["item_id"] for e in self.frozen.bridge_log if not e["bridged"]}
        # I03/I06/I10 跨音频难度差异显著，桥接必须留痕说明原因。
        self.assertTrue({"I03", "I06", "I10"} <= failed)
        for entry in self.frozen.bridge_log:
            if not entry["bridged"]:
                self.assertIn("reason", entry)

    def test_single_mode_under_calibrated_item_creates_no_bridge(self):
        # I14 仅支持 large_print 且只有 4 条记录：单呈现方式题无法搭桥，
        # 且在审计记录中不应出现伪桥接。
        bridge_ids = {e["item_id"] for e in self.frozen.bridge_log
                      if e["bridged"]}
        self.assertNotIn("I14", bridge_ids)
        self.assertNotIn("I13", bridge_ids)

    def test_bridge_threshold_constant_is_tight(self):
        self.assertGreater(BRIDGE_MAX_DIFF, 0.0)
        self.assertLess(BRIDGE_MAX_DIFF, 0.2)

    def test_zero_sample_item_uses_family_fallback_prior(self):
        # I26 是新增音频题、无任何统计记录。
        family_id = self.frozen.family_for_mode("audio")
        alpha, beta = self.frozen.fallback_posterior(family_id)
        self.assertGreater(alpha, 0.0)
        self.assertGreater(beta, 0.0)
        # 弱信息先验，不会制造虚假的确定性。
        self.assertLess(alpha + beta - 2.0, 10.0)


if __name__ == "__main__":
    unittest.main()

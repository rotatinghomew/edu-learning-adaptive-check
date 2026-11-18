import unittest
from pathlib import Path

from learning_adaptive_check.bank import QuestionBank, CONTENT_INVALID, NORMAL_RETIREMENT
from learning_adaptive_check.calibration import calibrate_bank
from learning_adaptive_check.records import load_skills, load_items, load_item_stats
from learning_adaptive_check.scales import decide_scale_families
from learning_adaptive_check.session import (
    AssessmentSession, handle_withdrawal, assess_skill,
    QUALIFIED, NOT_MASTERED, INCONCLUSIVE)

DATA = Path(__file__).parent.parent / "data"


def build_bank(withdrawals=None):
    skills = load_skills(DATA / "skills.csv")
    items = load_items(DATA / "items.csv", skills)
    stats = load_item_stats(DATA / "item_stats.csv", items)
    bank = QuestionBank(items, skills, calibrate_bank(stats),
                        decide_scale_families(items, calibrate_bank(stats)))
    for item_id, (reason, kind) in (withdrawals or {}).items():
        bank.withdraw(item_id, reason, kind)
    return bank, skills


class SkillAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.skill = {"skill_id": "S1", "name": "x",
                      "required_evidence": 3, "mastery": 0.6}

    def test_three_states_and_evidence_gate(self):
        inconclusive = assess_skill(self.skill, [(True, 1.0), (True, 1.0)])
        qualified = assess_skill(self.skill, [(True, 1.0)] * 4)
        not_mastered = assess_skill(self.skill, [(False, 1.0)] * 3)
        self.assertEqual(inconclusive["status"], INCONCLUSIVE)
        self.assertEqual(qualified["status"], QUALIFIED)
        self.assertEqual(not_mastered["status"], NOT_MASTERED)

    def test_enough_evidence_but_ambiguous_stays_inconclusive(self):
        # 2 对 2 错：证据量达标但两侧概率都不够高，不能武断下结论。
        result = assess_skill(
            self.skill, [(True, 1.0), (True, 1.0),
                         (False, 1.0), (False, 1.0)])
        self.assertEqual(result["status"], INCONCLUSIVE)
        self.assertIn("不确定", result["reason"])

    def test_low_weight_answers_cannot_shortcut_qualification(self):
        # 小样本题权重折减：三道半权题全对也不满足证据量门槛。
        result = assess_skill(self.skill, [(True, 0.5)] * 3)
        self.assertEqual(result["status"], INCONCLUSIVE)
        self.assertLess(result["total_weight"], self.skill["required_evidence"])

    def test_every_state_has_explanation(self):
        for answers in ([], [(True, 1.0)] * 4, [(False, 1.0)] * 3):
            self.assertTrue(assess_skill(self.skill, answers)["reason"])


class SessionRulesTests(unittest.TestCase):
    def setUp(self):
        self.bank, self.skills = build_bank()
        self.audio_family = self.bank.scales.family_for_mode("audio")
        self.text_family = self.bank.scales.family_for_mode("text")

    def test_session_rejects_mode_outside_family(self):
        with self.assertRaisesRegex(ValueError, "不能在该尺度下开测"):
            AssessmentSession("s1", "L", self.bank, self.audio_family, "text")

    def test_submit_rejects_cross_family_item(self):
        session = AssessmentSession(
            "s2", "L", self.bank, self.audio_family, "audio",
            target_skills=["S4"])
        # S4 没有任何音频题：跨族题不能拼进来凑覆盖。
        item_id, why = session.next_item()
        self.assertIsNone(item_id)
        self.assertIn("已耗尽", why)
        # 族外题（I11 仅文字、属文字尺度族）不能提交，不能拼用互不兼容的尺度。
        accepted, code, explain = session.submit("I11", True)
        self.assertFalse(accepted)
        self.assertEqual(code, "MODE_UNSUPPORTED")
        self.assertIn("audio", explain)

    def test_scale_isolation_is_enforced_structurally(self):
        # 尺度不兼容在两个层面被挡住：
        # 1) 会话构造期：学员呈现方式不属于目标尺度族时直接拒绝开测；
        # 2) 候选题层：eligible_items 永不返回族外题，提交侧再留防御性拒绝。
        with self.assertRaisesRegex(ValueError, "不能在该尺度下开测"):
            AssessmentSession("s-iso", "L", self.bank,
                              self.audio_family, "text")
        eligible = self.bank.eligible_items(self.audio_family, "audio")
        for item_id in eligible:
            self.assertEqual(
                self.bank.scales.family_of_item_mode(item_id, "audio"),
                self.audio_family)
        # 文字专属题不会出现在音频族候选中。
        text_only = [i for i, item in self.bank.items.items()
                     if item["modes"] == ("text",)]
        self.assertTrue(text_only)
        self.assertTrue(set(text_only).isdisjoint(eligible))
        # 提交侧防御分支仍保留：提交族外题得到拒绝码而非脏证据。
        session = AssessmentSession(
            "s-iso2", "L", self.bank, self.audio_family, "audio",
            target_skills=["S1"])
        accepted, code, _ = session.submit("I13", True)  # 仅文字的 S1 题
        self.assertFalse(accepted)
        self.assertIn(code, {"MODE_UNSUPPORTED", "SCALE_MISMATCH"})

    def test_duplicate_submission_is_idempotent(self):
        session = AssessmentSession(
            "s3", "L", self.bank, self.audio_family, "audio",
            target_skills=["S1"])
        item_id, _ = session.next_item()
        ok1, _, _ = session.submit(item_id, True)
        self.assertTrue(ok1)
        ok2, code, explain = session.submit(item_id, False)
        self.assertFalse(ok2)
        self.assertEqual(code, "DUPLICATE_REJECTED")
        self.assertIn("重复提交", explain)
        # 证据只有一份，不会被第二次提交改写。
        board = session.skill_board()
        self.assertEqual(board["S1"]["correct_weight"], 1.0)
        self.assertEqual(board["S1"]["wrong_weight"], 0.0)

    def test_exhausted_items_end_inconclusive_never_pass(self):
        # S4 无音频题：会话只能以证据不足结束。
        session = AssessmentSession(
            "s4", "L", self.bank, self.audio_family, "audio",
            target_skills=["S4"])
        code, explain = session.close()
        self.assertEqual(code, "CLOSED_INCONCLUSIVE")
        self.assertIn("不存在题目耗尽即通过", explain)
        self.assertEqual(session.skill_board()["S4"]["status"], INCONCLUSIVE)

    def test_selection_covers_all_target_skills_before_drilling(self):
        session = AssessmentSession(
            "s5", "L", self.bank, self.audio_family, "audio",
            target_skills=["S1", "S2", "S3"])
        seen_skills = []
        for _ in range(3):
            item_id, _ = session.next_item()
            self.assertIsNotNone(item_id)
            session.submit(item_id, True)
            seen_skills.append(self.bank.items[item_id]["skill_id"])
        # 前三题必须分别覆盖三个目标技能，而不是在同一技能上连出。
        self.assertEqual(sorted(seen_skills), ["S1", "S2", "S3"])

    def test_exposure_penalty_rotates_familiar_items(self):
        # 两道信息价值相同的题，历史曝光高的应被排到后面。
        session_a = AssessmentSession(
            "s6a", "LA", self.bank, self.text_family, "text",
            target_skills=["S4"], usage_counts={})
        first_a, _ = session_a.next_item()
        session_b = AssessmentSession(
            "s6b", "LB", self.bank, self.text_family, "text",
            target_skills=["S4"], usage_counts={first_a: 5})
        first_b, _ = session_b.next_item()
        self.assertNotEqual(first_a, first_b)
        # 轮换后再选仍在合法候选范围内。
        self.assertIn(first_b, self.bank.eligible_items(
            self.text_family, "text", "S4"))

    def test_every_selection_and_stop_is_explained(self):
        session = AssessmentSession(
            "s7", "L", self.bank, self.audio_family, "audio",
            target_skills=["S1"])
        item_id, why = session.next_item()
        self.assertIn("优先补技能", why)
        session.submit(item_id, True)
        session.next_item()
        codes = {event["code"] for event in session.events}
        self.assertIn("ITEM_SELECTED", codes)
        self.assertTrue(all(event["explanation"] for event in session.events))

    def test_closed_session_rejects_further_answers(self):
        session = AssessmentSession(
            "s8", "L", self.bank, self.text_family, "text",
            target_skills=["S4"])
        item_id, _ = session.next_item()
        session.submit(item_id, True)
        session.close()
        accepted, code, _ = session.submit(item_id, False)
        self.assertFalse(accepted)
        self.assertEqual(code, "SESSION_CLOSED")
        next_item, _ = session.next_item()
        self.assertIsNone(next_item)

    def test_decided_skill_never_receives_more_items(self):
        # S2 判未掌握后，即使该技能还有未作答题，也不能再对它出题。
        session = AssessmentSession(
            "s-decide", "L", self.bank, self.audio_family, "audio",
            target_skills=["S1", "S2"])
        answered = set()
        while True:
            item_id, why = session.next_item()
            if item_id is None:
                break
            skill = self.bank.items[item_id]["skill_id"]
            # S2 全错、S1 全对。
            session.submit(item_id, skill == "S1")
            answered.add(item_id)
            board = session.skill_board()
            if board["S2"]["status"] == NOT_MASTERED:
                later_skills = set()
                while True:
                    nxt, _ = session.next_item()
                    if nxt is None:
                        break
                    later_skills.add(self.bank.items[nxt]["skill_id"])
                    session.submit(nxt, True)
                self.assertNotIn("S2", later_skills)
                return
        self.fail("S2 未在题尽前形成未掌握判断")

    def test_selection_falls_through_when_top_deficit_skill_exhausted(self):
        # 撤光 S1 音频题：S1 缺口最大但无候选，选题必须顺延到 S2 而不是停止。
        bank, _ = build_bank()
        family = bank.scales.family_for_mode("audio")
        for item_id in bank.eligible_items(family, "audio", "S1"):
            bank.withdraw(item_id, "停用", NORMAL_RETIREMENT)
        session = AssessmentSession(
            "s-fall", "L", bank, family, "audio",
            target_skills=["S1", "S2"])
        item_id, why = session.next_item()
        self.assertIsNotNone(item_id)
        self.assertEqual(bank.items[item_id]["skill_id"], "S2")
        self.assertIn("顺延", why)
        # S1 仍然挂起：题尽技能最终只能是证据不足。
        code, _ = session.close()
        self.assertEqual(code, "CLOSED_INCONCLUSIVE")

    def test_unsupported_mode_and_unknown_item_rejected(self):
        session = AssessmentSession(
            "s9", "L", self.bank, self.audio_family, "audio",
            target_skills=["S1"])
        accepted, code, _ = session.submit("I13", True)  # 仅文字
        self.assertFalse(accepted)
        self.assertEqual(code, "MODE_UNSUPPORTED")
        accepted, code, _ = session.submit("NOPE", True)
        self.assertFalse(accepted)
        self.assertEqual(code, "UNKNOWN_ITEM")


class WithdrawalTests(unittest.TestCase):
    def setUp(self):
        self.bank, self.skills = build_bank()
        self.audio_family = self.bank.scales.family_for_mode("audio")

    def test_content_invalid_withdrawal_removes_prior_evidence(self):
        session = AssessmentSession(
            "w1", "L", self.bank, self.audio_family, "audio",
            target_skills=["S3"])
        item_id, _ = session.next_item()
        session.submit(item_id, True)
        self.assertGreater(session.skill_board()["S3"]["total_weight"], 0)
        result = handle_withdrawal(session, item_id, "内容失效", CONTENT_INVALID)
        self.assertEqual(result["excluded"], [item_id])
        self.assertEqual(session.skill_board()["S3"]["total_weight"], 0.0)
        self.assertEqual(session.skill_board()["S3"]["status"], INCONCLUSIVE)
        self.assertFalse(self.bank.is_available(item_id))
        # 已撤下的题不能再收作答。
        accepted, code, _ = session.submit(item_id, True)
        self.assertFalse(accepted)
        self.assertEqual(code, "ITEM_WITHDRAWN")
        self.assertTrue(any(e["code"] == "WITHDRAWAL_EVIDENCE_REMOVED"
                            for e in session.events))

    def test_normal_retirement_keeps_prior_evidence(self):
        session = AssessmentSession(
            "w2", "L", self.bank, self.audio_family, "audio",
            target_skills=["S1"])
        item_id, _ = session.next_item()
        session.submit(item_id, True)
        handle_withdrawal(session, item_id, "版本更新", NORMAL_RETIREMENT)
        board = session.skill_board()
        self.assertEqual(board["S1"]["correct_weight"], 1.0)
        self.assertTrue(any(e["code"] == "WITHDRAWAL_EVIDENCE_KEPT"
                            for e in session.events))

    def test_withdrawal_before_any_answer_only_blocks_future_use(self):
        session = AssessmentSession(
            "w3", "L", self.bank, self.audio_family, "audio",
            target_skills=["S1"])
        item_id, _ = session.next_item()
        handle_withdrawal(session, item_id, "版权到期", NORMAL_RETIREMENT)
        next_item, _ = session.next_item()
        self.assertNotEqual(next_item, item_id)
        self.assertTrue(any(e["code"] == "WITHDRAWAL_NO_PRIOR_ANSWER"
                            for e in session.events))

    def test_key_evidence_withdrawal_forces_reevaluation_of_finish(self):
        # 先在文字尺度下凑够 S3 合格证据（音频题数量不足以演示本路径），
        # 再撤掉其中一道：必须退回证据不足，不允许带着已失效证据结束测查
        # 或给出选课资格。
        text_family = self.bank.scales.family_for_mode("text")
        session = AssessmentSession(
            "w4", "L", self.bank, text_family, "text",
            target_skills=["S3"])
        answered = []
        for _ in range(3):
            item_id, _ = session.next_item()
            session.submit(item_id, True)
            answered.append(item_id)
        self.assertEqual(session.skill_board()["S3"]["status"], QUALIFIED)
        handle_withdrawal(session, answered[0], "内容失效", CONTENT_INVALID)
        self.assertEqual(session.skill_board()["S3"]["status"], INCONCLUSIVE)
        code, _ = session.close()
        self.assertEqual(code, "CLOSED_INCONCLUSIVE")

    def test_withdrawn_item_never_selected_in_new_session(self):
        bank, _ = build_bank(withdrawals={"I04": ("停用", CONTENT_INVALID)})
        session = AssessmentSession(
            "w5", "L", bank, bank.scales.family_for_mode("audio"),
            "audio", target_skills=["S1"])
        for _ in range(6):
            item_id, _ = session.next_item()
            if item_id is None:
                break
            self.assertNotEqual(item_id, "I04")
            session.submit(item_id, True)


if __name__ == "__main__":
    unittest.main()

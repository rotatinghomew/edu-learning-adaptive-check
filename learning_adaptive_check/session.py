"""自适应测查会话。

关键规则：

* 一次会话绑定一个尺度族，族外题一律不出（禁止拼用互不兼容的尺度）；
* 技能状态三态：合格 QUALIFIED / 未掌握 NOT_MASTERED / 证据不足 INCONCLUSIVE，
  用带先验的 Beta 后验单侧概率判定，而不是固定题单式总分；
* 选题先消技能覆盖缺口（哪个技能离“可判断”最远先补哪个），再在该技能内
  按期望信息量排序，并对已曝光题降权，避免少数熟题反复出现；
* 作答提交幂等：同一 (会话, 题目) 重复提交不产生新证据；
* 题目耗尽只允许“证据不足”收尾，绝不自动判通过；
* 撤题由 handle_withdrawal 协同：内容失效剔除该题全部证据，正常下架保留；
  重算所有受影响技能并记录先前回答是否仍可用。
"""
from .calibration import betainc_regularized, PRIOR_CORRECT, PRIOR_WRONG
from .bank import CONTENT_INVALID, NORMAL_RETIREMENT

QUALIFIED = "qualified"
NOT_MASTERED = "not_mastered"
INCONCLUSIVE = "inconclusive"

MASTERY_PROB = 0.85        # P(掌握率 >= 技能掌握线) 达到该值才算有把握
NOT_MASTERY_PROB = 0.85    # P(掌握率 < 掌握线) 达到该值才算有把握判未掌握
EXPOSURE_PENALTY = 0.4     # 每曝光一次对选题分的折减
MIN_SCORE = 0.05


def assess_skill(skill, weighted_answers):
    """按带权证据计算单个技能的三态判断。

    weighted_answers: [(correct: bool, weight: float), ...]
    """
    correct_weight = sum(w for ok, w in weighted_answers if ok)
    wrong_weight = sum(w for ok, w in weighted_answers if not ok)
    total_weight = correct_weight + wrong_weight
    alpha = PRIOR_CORRECT + correct_weight
    beta = PRIOR_WRONG + wrong_weight
    threshold = skill["mastery"]
    p_above = 1.0 - betainc_regularized(alpha, beta, threshold)
    p_below = betainc_regularized(alpha, beta, threshold)
    required = skill["required_evidence"]

    if total_weight + 1e-12 < required:
        status = INCONCLUSIVE
        reason = (f"有效证据量 {total_weight:.2f} 低于要求 {required}，"
                 f"无法形成有把握的判断")
    elif p_above >= MASTERY_PROB:
        status = QUALIFIED
        reason = (f"证据量 {total_weight:.2f} 达标，且掌握率不低于 {threshold:.0%} "
                 f"的后验概率为 {p_above:.1%}")
    elif p_below >= NOT_MASTERY_PROB:
        status = NOT_MASTERED
        reason = (f"证据量 {total_weight:.2f} 达标，但掌握率低于 {threshold:.0%} "
                 f"的后验概率为 {p_below:.1%}")
    else:
        status = INCONCLUSIVE
        reason = (f"证据量 {total_weight:.2f} 达标，但单侧后验概率 "
                 f"{p_above:.1%}/{p_below:.1%} 均未越过 {MASTERY_PROB:.0%} 判据，"
                 f"判断仍不确定")
    return {
        "status": status,
        "reason": reason,
        "total_weight": round(total_weight, 4),
        "correct_weight": round(correct_weight, 4),
        "wrong_weight": round(wrong_weight, 4),
        "p_above": round(p_above, 4),
        "p_below": round(p_below, 4),
    }


def expected_information(low, high):
    """区间越宽，当前能力判断越不确定，该题信息价值越高。"""
    return max(MIN_SCORE, float(high) - float(low))


class AssessmentSession:
    def __init__(self, session_id, learner_id, bank, family_id, learner_mode,
                 target_skills=None, started_at=None, usage_counts=None):
        if learner_mode not in bank.scales.family_modes(family_id):
            raise ValueError(
                f"尺度族 {family_id} 不覆盖呈现方式 {learner_mode}，"
                f"不能在该尺度下开测（可用呈现方式: "
                f"{'、'.join(bank.scales.family_modes(family_id))}）")
        self.session_id = session_id
        self.learner_id = learner_id
        self.bank = bank
        self.family_id = family_id
        self.learner_mode = learner_mode
        self.target_skills = tuple(target_skills or sorted(bank.skills))
        unknown = set(self.target_skills) - set(bank.skills)
        if unknown:
            raise ValueError(f"未知技能: {', '.join(sorted(unknown))}")
        self.started_at = started_at
        self.closed = False
        # 跨会话累计曝光次数（题库层面的熟题轮换依据）。
        self.usage_counts = dict(usage_counts or {})
        # item_id -> {"correct": bool, "weight": float, "excluded": bool,
        #             "exclusion_reason": str|None}
        self.responses = {}
        self.events = []
        self._record("SESSION_START",
                     f"会话绑定尺度族 {family_id}（{bank.scales.families[family_id]['scope']}），"
                     f"呈现方式 {learner_mode}")

    def _record(self, code, explanation):
        self.events.append({"code": code, "explanation": explanation})

    # ---- 证据聚合 ------------------------------------------------------
    def _skill_answers(self, skill_id):
        result = []
        for item_id, response in self.responses.items():
            item = self.bank.items[item_id]
            if item["skill_id"] != skill_id or response["excluded"]:
                continue
            result.append((response["correct"], response["weight"]))
        return result

    def skill_board(self):
        board = {}
        for skill_id in self.target_skills:
            assessment = assess_skill(self.bank.skills[skill_id],
                                      self._skill_answers(skill_id))
            board[skill_id] = assessment
        return board

    def _coverage_deficit(self, skill_id):
        """覆盖缺口：距“可判断”所需的证据量，越大越优先补。

        已定案（合格/未掌握）的技能返回 -1，绝不参与后续选题；
        只有证据不足的技能才有非负缺口。
        """
        assessment = assess_skill(self.bank.skills[skill_id],
                                  self._skill_answers(skill_id))
        if assessment["status"] != INCONCLUSIVE:
            return -1.0
        required = self.bank.skills[skill_id]["required_evidence"]
        return max(0.0, required - assessment["total_weight"])

    # ---- 选题 ----------------------------------------------------------
    def next_item(self):
        """返回 (item_id, 解释)；无合适题时返回 (None, 解释)。"""
        if self.closed:
            return None, "会话已结束，不再出题"
        board = self.skill_board()
        pending = [sid for sid in self.target_skills
                   if board[sid]["status"] == INCONCLUSIVE]
        if not pending:
            return None, "所有目标技能均已有有把握的判断，无需继续出题"

        deficits = {sid: self._coverage_deficit(sid) for sid in pending}
        # 覆盖优先：按覆盖缺口从大到小处理技能；某技能候选耗尽时顺延到
        # 下一个技能，只有所有不确定技能都无题可出时才以证据不足停止。
        skill_order = sorted(pending, key=lambda sid: (-deficits[sid], sid))

        used = set(self.responses)
        scored = []
        chosen_skill = None
        skipped_skills = []
        for skill_id in skill_order:
            candidates = [item_id for item_id in self.bank.eligible_items(
                self.family_id, self.learner_mode, skill_id)
                if item_id not in used]
            if not candidates:
                skipped_skills.append(skill_id)
                continue
            chosen_skill = skill_id
            for item_id in candidates:
                low, high, _ = self.bank.posterior(item_id, self.learner_mode)
                info = expected_information(low, high)
                exposure_penalty = EXPOSURE_PENALTY * self._global_usage(item_id)
                score = info - exposure_penalty
                scored.append((item_id, skill_id, round(score, 4),
                               round(info, 4)))
            break
        if chosen_skill is None:
            return None, ("候选题目已耗尽：剩余不确定技能 "
                          f"{'、'.join(sorted(pending))} 在尺度族 "
                          f"{self.family_id}、呈现方式 {self.learner_mode} "
                          "下已无未作答且未撤下的题；这些技能只能按证据不足处理，"
                          "不会自动判为通过")
        # 同技能候选题按得分排序，并列时取 item_id 较小者，保证可复现。
        scored.sort(key=lambda entry: (-entry[2], entry[0]))
        top = scored[0]
        if skipped_skills:
            layer_note = ("更高覆盖缺口技能 "
                          f"{'、'.join(skipped_skills)} 候选题已耗尽，顺延至此；"
                          f"本技能缺口 {deficits[chosen_skill]:.2f}")
        else:
            layer_note = f"覆盖缺口 {deficits[chosen_skill]:.2f} 为当前最优先"
        explanation = (
            f"优先补技能 {chosen_skill}（{layer_note}）；题 {top[0]} 期望信息量 "
            f"{top[3]}，曝光折减后得分 {top[2]}，在该技能候选题中最高")
        self._record("ITEM_SELECTED", explanation)
        return top[0], explanation

    def _global_usage(self, item_id):
        """题库层面的累计曝光：会话内使用 + 历史会话使用。"""
        return self.usage_counts.get(item_id, 0) + (
            1 if item_id in self.responses else 0)

    # ---- 作答 ----------------------------------------------------------
    def submit(self, item_id, correct):
        """幂等提交。返回 (accepted, code, explanation)。"""
        if self.closed:
            return False, "SESSION_CLOSED", "会话已结束，作答不再接收"
        if not isinstance(correct, bool):
            return False, "BAD_ANSWER", "correct 必须是布尔值"
        if item_id not in self.bank.items:
            return False, "UNKNOWN_ITEM", f"题目 {item_id} 不存在"
        item = self.bank.items[item_id]
        if item["skill_id"] not in self.target_skills:
            return False, "OUT_OF_SCOPE", f"题目 {item_id} 不属于本会话目标技能"
        if self.learner_mode not in item["modes"]:
            return False, "MODE_UNSUPPORTED", (
                f"题目 {item_id} 不支持本学员的呈现方式 {self.learner_mode}")
        if self.bank.scales.family_of_item_mode(item_id, self.learner_mode) \
                != self.family_id:
            return False, "SCALE_MISMATCH", (
                f"题目 {item_id} 属于另一尺度族，不能与本会话结果拼用")
        if not self.bank.is_available(item_id):
            entry = self.bank.withdrawal(item_id)
            return False, "ITEM_WITHDRAWN", (
                f"题目 {item_id} 已撤下（原因: {entry['reason']}），"
                "本会话不再接受该题作答")
        if item_id in self.responses:
            previous = self.responses[item_id]
            return False, "DUPLICATE_REJECTED", (
                f"题目 {item_id} 已在本会话提交过（结果: "
                f"{'正确' if previous['correct'] else '错误'}），"
                "重复提交不会被当作新增能力证据")

        weight = self.bank.weight(item_id, self.learner_mode)
        self.responses[item_id] = {
            "correct": correct, "weight": weight,
            "excluded": False, "exclusion_reason": None,
        }
        board = self.skill_board()
        assessment = board[item["skill_id"]]
        self._record("ANSWER_ACCEPTED",
                     f"题 {item_id} 作答{'正确' if correct else '错误'}，"
                     f"证据权重 {weight:.2f}；技能 {item['skill_id']} 当前状态: "
                     f"{assessment['status']}（{assessment['reason']}）")
        return True, "ACCEPTED", "作答已计入"

    # ---- 结束 ----------------------------------------------------------
    def close(self):
        board = self.skill_board()
        inconclusive = [sid for sid, a in board.items()
                        if a["status"] == INCONCLUSIVE]
        self.closed = True
        if inconclusive:
            code, explanation = "CLOSED_INCONCLUSIVE", (
                "测查结束但证据不足，未对以下技能做通过判断: "
                f"{'、'.join(sorted(inconclusive))}；需要补充测查，"
                "不存在题目耗尽即通过的情况")
        else:
            code, explanation = "CLOSED_COMPLETE", (
                "所有目标技能均已形成有把握的合格/未掌握判断，测查结束")
        self._record(code, explanation)
        return code, explanation


def handle_withdrawal(session, item_id, reason, kind=CONTENT_INVALID):
    """测查进行期间撤题的协同处理。"""
    if item_id not in session.bank.items:
        raise KeyError(f"未知题目: {item_id}")
    session.bank.withdraw(item_id, reason, kind)
    affected_skills = {session.bank.items[item_id]["skill_id"]}
    if item_id not in session.responses:
        session._record(
            "WITHDRAWAL_NO_PRIOR_ANSWER",
            f"题 {item_id} 撤下（{reason}）；本会话尚未使用该题，"
            "无历史回答需要处理，后续选题自动跳过该题")
        return {"affected_skills": sorted(affected_skills), "excluded": []}

    response = session.responses[item_id]
    excluded = []
    if kind == CONTENT_INVALID:
        response["excluded"] = True
        response["exclusion_reason"] = reason
        excluded.append(item_id)
        session._record(
            "WITHDRAWAL_EVIDENCE_REMOVED",
            f"题 {item_id} 因内容失效撤下（{reason}）；该题先前作答"
            f"（{'正确' if response['correct'] else '错误'}）不再作为能力证据，"
            "相关技能状态已重算，是否能结束测查与选课资格随之重新判断")
    else:
        session._record(
            "WITHDRAWAL_EVIDENCE_KEPT",
            f"题 {item_id} 正常下架（{reason}）；撤题前的作答仍然有效并保留，"
            "只是后续不再出该题")

    board = session.skill_board()
    for skill_id in sorted(affected_skills):
        assessment = board[skill_id]
        session._record(
            "SKILL_REEVALUATED",
            f"技能 {skill_id} 撤题后重算: {assessment['status']}"
            f"（{assessment['reason']}）")
    return {"affected_skills": sorted(affected_skills),
            "excluded": excluded,
            "kind": kind}

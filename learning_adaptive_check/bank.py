"""题库门面：题目、撤题状态与尺度族内的可用性查询。"""
from .calibration import calibrate_mode, evidence_weight

CONTENT_INVALID = "content_invalid"   # 内容失效：已作答证据也要剔除
NORMAL_RETIREMENT = "normal"          # 正常下架：已作答证据保留


class QuestionBank:
    def __init__(self, items, skills, calibration, scale_frozen):
        self.items = dict(items)
        self.skills = dict(skills)
        self.calibration = dict(calibration)
        self.scales = scale_frozen
        # item_id -> {"reason": str, "kind": CONTENT_INVALID/NORMAL}
        self.withdrawals = {}
        self._skill_items = {}
        for item_id, item in self.items.items():
            self._skill_items.setdefault(item["skill_id"], []).append(item_id)

    # ---- 撤题 ----------------------------------------------------------
    def withdraw(self, item_id, reason, kind=CONTENT_INVALID):
        if item_id not in self.items:
            raise KeyError(f"未知题目: {item_id}")
        if kind not in (CONTENT_INVALID, NORMAL_RETIREMENT):
            raise ValueError(f"未知撤题类型: {kind}")
        if item_id in self.withdrawals:
            raise ValueError(f"题目 {item_id} 已撤下，不能重复撤题")
        self.withdrawals[item_id] = {"reason": reason, "kind": kind}

    def is_available(self, item_id):
        return item_id in self.items and item_id not in self.withdrawals

    def withdrawal(self, item_id):
        return self.withdrawals.get(item_id)

    def skill_item_ids(self, skill_id):
        return tuple(sorted(self._skill_items.get(skill_id, ())))

    # ---- 尺度族内的题目统计 --------------------------------------------
    def posterior(self, item_id, mode):
        """题目在某呈现方式下的 Beta 后验；零样本时借用所属族先验。"""
        cal = self.calibration.get((item_id, mode))
        if cal is not None:
            return cal["credible_low"], cal["credible_high"], \
                cal["posterior_mean"]
        family_id = self.scales.family_of_item_mode(item_id, mode)
        alpha, beta = self.scales.fallback_posterior(family_id)
        mean = alpha / (alpha + beta)
        return mean, mean, mean

    def weight(self, item_id, mode):
        cal = self.calibration.get((item_id, mode))
        if cal is not None:
            return cal["evidence_weight"]
        return evidence_weight(0)

    def calibrated(self, item_id, mode):
        cal = self.calibration.get((item_id, mode))
        return bool(cal and cal["calibrated"])

    # ---- 选题候选 ------------------------------------------------------
    def eligible_items(self, family_id, learner_mode, skill_id=None):
        """学员在指定尺度族与无障碍呈现方式下仍可作答的题。"""
        result = []
        item_ids = (self.skill_item_ids(skill_id) if skill_id
                    else sorted(self.items))
        for item_id in item_ids:
            item = self.items[item_id]
            if learner_mode not in item["modes"]:
                continue
            if self.scales.family_of_item_mode(item_id, learner_mode) != family_id:
                continue
            if not self.is_available(item_id):
                continue
            result.append(item_id)
        return tuple(result)

    def family_summary(self):
        return [
            {"family_id": fam["family_id"], "modes": fam["modes"],
             "scope": fam["scope"],
             "pooled_prior_mean": round(fam["pooled_prior_mean"], 4)}
            for fam in sorted(self.scales.families.values(),
                              key=lambda f: f["family_id"])
        ]

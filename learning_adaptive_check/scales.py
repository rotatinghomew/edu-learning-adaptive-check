"""跨呈现方式的尺度族判定。

业务约束：跨呈现方式的结果能否共用同一尺度必须有证据，并给出适用范围。

规则：
* 同一道题在多个呈现方式下都达到校准样本量，且各呈现方式后验均值
  最大差异不超过 BRIDGE_MAX_DIFF，才构成一座“桥”，允许合并这些呈现方式；
* 桥构成的连通分量形成一个尺度族，族内呈现方式共用尺度；
* 无法搭桥的呈现方式（例如音频与文字难度差异显著）独立成族；
* 一次测查只能绑定一个尺度族，族外题的作答不能与族内结果拼用。

判定过程逐条留痕（通过/不通过及原因），供报告解释适用范围。
"""
from .calibration import calibrate_mode

BRIDGE_MAX_DIFF = 0.1        # 桥接题跨呈现方式后验均值最大允许差异
FALLBACK_PRIOR_STRENGTH = 4  # 族内零样本新题借用族先验的等效样本量


class _UnionFind:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}

    def find(self, node):
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            node, self.parent[node] = self.parent[node], root
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def decide_scale_families(items, calibration):
    """返回 ScaleFrozen 结构：尺度族、题-呈现方式归属、桥接审计记录。"""
    modes = sorted({mode for item in items.values() for mode in item["modes"]})
    uf = _UnionFind(modes)
    bridge_log = []

    for item_id, item in sorted(items.items()):
        item_modes = list(item["modes"])
        if len(item_modes) < 2:
            continue
        per_mode = {}
        reasons = []
        all_calibrated = True
        for mode in item_modes:
            cal = calibration.get((item_id, mode)) or calibrate_mode(0, 0)
            per_mode[mode] = cal
            if not cal["calibrated"]:
                all_calibrated = False
                reasons.append(f"{mode} 样本量 {cal['attempts']} 未达校准要求")
        if not all_calibrated:
            bridge_log.append({"item_id": item_id, "bridged": False,
                               "reason": "；".join(reasons)})
            continue
        max_diff = max(abs(per_mode[a]["posterior_mean"]
                           - per_mode[b]["posterior_mean"])
                       for i, a in enumerate(item_modes)
                       for b in item_modes[i + 1:])
        if max_diff > BRIDGE_MAX_DIFF:
            detail = "，".join(f"{m}={per_mode[m]['posterior_mean']:.3f}"
                               for m in item_modes)
            bridge_log.append({"item_id": item_id, "bridged": False,
                               "reason": f"后验均值差异 {max_diff:.3f} 超过阈值 "
                                         f"{BRIDGE_MAX_DIFF}（{detail}）"})
            continue
        for mode in item_modes[1:]:
            uf.union(item_modes[0], mode)
        bridge_log.append({"item_id": item_id, "bridged": True,
                           "modes": tuple(item_modes),
                           "max_diff": round(max_diff, 4),
                           "reason": "各呈现方式均已校准且难度差异在阈值内"})

    components = {}
    for mode in modes:
        components.setdefault(uf.find(mode), []).append(mode)

    families = {}
    mode_family = {}
    for member_modes in components.values():
        member_modes = sorted(member_modes)
        family_id = "+".join(member_modes)
        pooled_values = [
            cal["posterior_mean"]
            for (item_id, mode), cal in calibration.items()
            if mode in member_modes and cal["calibrated"]
        ]
        pooled_mean = sum(pooled_values) / len(pooled_values) if pooled_values else 0.5
        bridges = [entry["item_id"] for entry in bridge_log
                   if entry["bridged"] and set(entry["modes"]) <= set(member_modes)]
        if len(member_modes) > 1:
            scope = (f"共用尺度（呈现方式: {'、'.join(member_modes)}；"
                     f"桥接题 {len(bridges)} 道: {'、'.join(bridges)}）")
        else:
            scope = (f"独立尺度（呈现方式: {member_modes[0]}；"
                     f"无充分桥接证据支持与其他呈现方式共用）")
        families[family_id] = {
            "family_id": family_id,
            "modes": tuple(member_modes),
            "pooled_prior_mean": pooled_mean,
            "bridge_items": tuple(bridges),
            "scope": scope,
        }
        for mode in member_modes:
            mode_family[mode] = family_id

    item_mode_family = {}
    for item_id, item in items.items():
        for mode in item["modes"]:
            item_mode_family[(item_id, mode)] = mode_family[mode]

    return ScaleFrozen(families, mode_family, item_mode_family, bridge_log)


class ScaleFrozen:
    """尺度判定的不可变结果，供题库与测查会话查询。"""

    def __init__(self, families, mode_family, item_mode_family, bridge_log):
        self.families = families
        self.mode_family = mode_family
        self.item_mode_family = item_mode_family
        self.bridge_log = bridge_log

    def family_for_mode(self, mode):
        return self.mode_family.get(mode)

    def family_of_item_mode(self, item_id, mode):
        return self.item_mode_family.get((item_id, mode))

    def fallback_posterior(self, family_id):
        """族内完全没有统计记录的新题：借用族先验，仍保持弱信息。"""
        mean = self.families[family_id]["pooled_prior_mean"]
        alpha = 1.0 + FALLBACK_PRIOR_STRENGTH * mean
        beta = 1.0 + FALLBACK_PRIOR_STRENGTH * (1.0 - mean)
        return alpha, beta

    def family_modes(self, family_id):
        return self.families[family_id]["modes"]

"""题库统计与校准。

判据（口径集中在此，供测试直接核对）：

* 每题每个呈现方式用 Beta-二项收缩估计正确率，样本越少先验影响越大，
  偶然高答对率不会让小样本题被当成容易题；
* 题目 ``calibrated`` 要求该呈现方式下样本量 >= MIN_ATTEMPTS，
  否则只能给出宽区间，标记为“校准不足”；
* 容易题/难题用单侧后验概率判定，而不是裸答对率；
* 证据权重 ``evidence_weight``：校准题 = 1.0，校准不足题按样本量线性
  折减到 NEW_ITEM_MIN_WEIGHT，避免少量记录的题与成熟题同权。
"""
import math

MIN_ATTEMPTS = 20          # 达到该样本量才视为校准题
PRIOR_CORRECT = 1.0        # Beta(1, 1) 均匀先验
PRIOR_WRONG = 1.0
EASY_PROB = 0.85           # P(正确率 >= EASY_THRESHOLD) 超过该值才算容易
EASY_THRESHOLD = 0.8
HARD_PROB = 0.85           # P(正确率 <= HARD_THRESHOLD) 超过该值才算难题
HARD_THRESHOLD = 0.4
NEW_ITEM_MIN_WEIGHT = 0.5  # 零样本新题的证据权重


def _betacf(a, b, x, max_iter=200, eps=3e-14):
    """连分式展开（Numerical Recipes 第 6 章），用于规则化不完全 Beta。"""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc_regularized(a, b, x):
    """I_x(a, b)：Beta(a,b) 下变量不超过 x 的累积概率。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_posterior(attempts, correct):
    """返回 Beta 后验参数 (alpha, beta)。"""
    return (PRIOR_CORRECT + correct, PRIOR_WRONG + attempts - correct)


def credible_interval(alpha, beta, confidence=0.9):
    """等尾可信区间。无第三方依赖，用二分反解 CDF。"""
    tail = (1.0 - confidence) / 2.0

    def invert(target):
        lo, hi = 1e-9, 1.0 - 1e-9
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if betainc_regularized(alpha, beta, mid) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    return invert(tail), invert(1.0 - tail)


def evidence_weight(attempts):
    """新题/小样本题的证据权重随样本量线性折减。"""
    if attempts >= MIN_ATTEMPTS:
        return 1.0
    return NEW_ITEM_MIN_WEIGHT + (1.0 - NEW_ITEM_MIN_WEIGHT) * attempts / MIN_ATTEMPTS


def calibrate_mode(attempts, correct):
    """单个 (题目, 呈现方式) 的校准结果。"""
    alpha, beta = beta_posterior(attempts, correct)
    mean = alpha / (alpha + beta)
    low, high = credible_interval(alpha, beta)
    calibrated = attempts >= MIN_ATTEMPTS
    # 单侧后验概率：样本不足时自然趋向 0.5，无法越过 0.85 判据。
    p_easy = 1.0 - betainc_regularized(alpha, beta, EASY_THRESHOLD)
    p_hard = betainc_regularized(alpha, beta, HARD_THRESHOLD)
    if calibrated and p_easy >= EASY_PROB:
        difficulty = "easy"
    elif calibrated and p_hard >= HARD_PROB:
        difficulty = "hard"
    else:
        difficulty = "undetermined"
    return {
        "attempts": attempts,
        "correct": correct,
        "raw_rate": correct / attempts if attempts else None,
        "posterior_mean": mean,
        "credible_low": low,
        "credible_high": high,
        "calibrated": calibrated,
        "difficulty": difficulty,
        "evidence_weight": evidence_weight(attempts),
        "p_easy": p_easy,
        "p_hard": p_hard,
    }


def calibrate_bank(stats_rows):
    """聚合全部统计行 -> {(item_id, mode): 校准结果}。"""
    return {
        (row["item_id"], row["mode"]):
            calibrate_mode(row["attempts"], row["correct"])
        for row in stats_rows
    }

# thresholds.py
# 幾何判斷門檻集中管理（方便日後微調）

OBLONG_R_LEN_CHEEK      = 1.20  # 臉長 / 顴骨寬 超過此值 → 長臉
SQUARE_R_JAW_CHEEK_MIN  = 0.90  # 下顎 / 顴骨寬 下限
SQUARE_R_JAW_CHEEK_MAX  = 1.10  # 下顎 / 顴骨寬 上限
SQUARE_JAW_ANGLE_MAX    = 165   # 方臉：角度需小於此（越銳越方）
ROUND_JAW_ANGLE_MIN     = 165   # 圓臉 / 鵝蛋臉：角度需大於此（越鈍越圓）
ROUND_LEN_CHEEK_DELTA   = 0.10  # 「長寬接近」定義（10% 內視為相等）

# 算分析可信度
def build_confidence_report(p1, p2, used_geo, geometry_info):
    """
    評估本次分析可信度。

    注意：
    這不是模型準確率，也不是數學上的百分比。
    它只是根據模型信心值、Top1/Top2 差距、Landmark、Geometry 狀態，
    給出一個好理解的可信度等級。
    """
    # p1 是模型對 Top1 的信心值，p2 是模型對 Top2 的信心值
    # used_geo 本次是否啟用 Geometry 補判
    # geometry_info = Geometry 分析的結果
    top_gap = max(p1 - p2, 0)

    if p1 >= 0.85 and top_gap >= 0.35 and geometry_info and not used_geo:
        level = "Excellent"
        label = "可信度高"
        stars = "★★★★★"
    elif p1 >= 0.70 and top_gap >= 0.20 and geometry_info:
        level = "Good"
        label = "可信度良好"
        stars = "★★★★☆"
    elif p1 >= 0.55:
        level = "Fair"
        label = "僅供參考"
        stars = "★★★☆☆"
    else:
        level = "Low"
        label = "可信度偏低"
        stars = "★★☆☆☆"

    messages = []

    if p1 >= 0.75:
        messages.append("模型信心值高，主要判斷較穩定。")
    elif p1 >= 0.55:
        messages.append("模型信心值中等，建議搭配機率分布一起判讀。")
    else:
        messages.append("模型信心值偏低，臉型結果需要保守參考。")

    if top_gap >= 0.30:
        messages.append("Top1 與 Top2 差距明顯，分類結果較穩定。")
    else:
        messages.append("Top1 與 Top2 接近，代表模型對臉型仍有不確定性。")

    if geometry_info:
        messages.append("Landmark 偵測成功，可提供輔助分析。")
    else:
        messages.append("未取得 Landmark，因此缺少幾何輔助資訊。")

    if used_geo:
        messages.append("本次啟用了 Geometry 補判，代表模型原始信心值較低。")
    else:
        messages.append("本次未啟用 Geometry 補判，代表模型信心值已達門檻。")

    return {
        "level": level,
        "label": label,
        "stars": stars,
        "model_confidence": p1,
        "top_gap": top_gap,
        "landmark_detected": geometry_info is not None,
        "geometry_used": used_geo,
        "messages": messages,
    }
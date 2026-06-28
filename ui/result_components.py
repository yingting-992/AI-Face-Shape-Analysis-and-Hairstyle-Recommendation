# 分析結果區的 HTML

# ui/result_components.py

def build_result_title(top1, p1, top2, p2, used_geo):
    """
    產生右上角「臉型判定結果」那張卡片。

    這個函式只負責畫面文字，
    不做模型判斷，也不碰資料處理。
    """
    html = (
        f"<div class='result-title-card'>"
        f"<h2>臉型判定結果：<strong>{top1}</strong></h2>"
        f"<p><b>模型信心值：</b>{p1:.2f}<br>"
        f"第二可能結果：{top2}（{p2:.2f}）</p>"
        f"</div>"
    )

    if used_geo:
        html += "<p>已啟用低信心幾何補判機制。</p>"

    return html


def feature_bar(label, desc, percent):
    """
    產生幾何輔助分析用的橫條。

    label：特徵名稱，例如臉部長度
    desc：文字判斷，例如偏長、適中、偏寬
    percent：橫條長度，0 到 100
    """
    return f"""
    <div style="margin-bottom:18px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:7px;">
            <span style="color:var(--text); font-weight:900;">▸ {label}</span>
            <span style="color:var(--primary-active); font-weight:900;">{desc}</span>
        </div>

        <div style="height:10px; background:#E8DDD8; border-radius:999px; overflow:hidden;">
            <div style="width:{percent}%; height:100%; background:var(--primary); border-radius:999px;"></div>
        </div>
    </div>
    """


def build_probability_bars(prob_dict):
    """
    把模型每個臉型的機率轉成橫條圖。

    prob_dict 長得像：
    {
        "Round": 0.83,
        "Oval": 0.12
    }
    """
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

    html = ""
    for cls_name, score in sorted_probs:
        pct = score * 100
        html += f"""
        <div style="margin-bottom: 12px;">
            <div style="display:flex; justify-content:space-between; font-size:14px; font-weight:900; color:var(--text);">
                <span style="color:var(--text);">{cls_name}</span>
                <span style="color:var(--text);">{pct:.1f}%</span>
            </div>
            <div style="height:10px; background:rgba(120,90,75,0.16); border-radius:999px; overflow:hidden; margin-top:6px;">
                <div style="width:{pct:.1f}%; height:100%; background:var(--primary); border-radius:999px;"></div>
            </div>
        </div>
        """

    return html

# 臉型組成分析
def build_composition_text(top1, p1, top2, p2):
    """
    產生「臉型組成分析」那段文字。

    白話講：
    這裡就是把 Top1、Top2 跟模型信心值，
    轉成使用者看得懂的說明。
    """
    return f"""
    <p style="color: var(--text); opacity: 1; line-height: 1.9; font-size: 15px; font-weight: 500;">
        本次系統判定主要臉型為 
        <strong style="color: var(--primary-active); font-weight: 900;">{top1}</strong>，
        模型信心值為 
        <strong style="color: var(--primary-active); font-weight: 900;">{p1*100:.1f}%</strong>。
        <br><br>

        第二可能臉型為 
        <strong style="color: var(--primary-active); font-weight: 900;">{top2}</strong>
        （<strong style="color: var(--primary-active); font-weight: 900;">{p2*100:.1f}%</strong>）。
        此結果代表模型偵測到部分 
        <strong style="color: var(--primary-active); font-weight: 900;">{top2}</strong>
        輪廓特徵，系統會將其列為參考資訊。
        <br><br>

        因此推薦主要依據 
        <strong style="color: var(--primary-active); font-weight: 900;">{top1}</strong>
        臉型設計原則產生，並可輔助參考 
        <strong style="color: var(--primary-active); font-weight: 900;">{top2}</strong>
        的修飾方向。
    </p>
    """

# 幾何輔助分析
def build_feature_analysis_html(geometry_info):
    """
    產生「幾何輔助分析」那一整塊 HTML。

    白話講：
    這裡不是要精準判斷臉型，
    而是把 FaceMesh 算出來的幾何數值整理給使用者看。

    為什麼不再硬說偏寬、偏窄？
    因為實測後發現，上臉寬度、下顎寬度這些比例在不同照片之間差距很小，
    不適合當成主要判斷依據。

    所以這一區的定位是：
    1. 輔助說明
    2. 提醒哪些特徵可信度較高
    3. 讓系統更可解釋，但不取代 ResNet34 分類結果
    """
    if not geometry_info:
        return """
        <p style="color:var(--text); line-height:1.8;">
            目前未取得臉部 Landmark，因此無法產生幾何輔助分析。
        </p>
        """

    face_len = geometry_info["face_length_ratio"]
    jaw_ratio = geometry_info["jaw_width_ratio"]
    upper_ratio = geometry_info["forehead_width_ratio"]
    jaw_angle = geometry_info["jaw_angle"]

    face_len_pct = min(max(face_len / 1.35 * 100, 8), 100)
    upper_pct = min(max(upper_ratio / 1.05 * 100, 8), 100)
    jaw_pct = min(max(jaw_ratio / 1.10 * 100, 8), 100)
    jaw_angle_pct = min(max(jaw_angle / 180 * 100, 8), 100)

    face_len_note = "可參考"
    upper_note = "低可信"
    jaw_note = "低可信"
    jaw_angle_note = "可參考"

    face_len_desc = "臉長比例較有參考價值，可輔助觀察長臉傾向。"
    upper_desc = "FaceMesh 不會抓到完整額頭或髮際線，因此上臉寬度只能當輔助參考。"
    jaw_desc = "實測後發現下顎寬度比例在不同臉型間差異偏小，不適合單獨判斷臉型。"
    jaw_angle_desc = "下顎角度可用來描述線條柔和或明顯，但仍可能受姿勢與拍攝角度影響。"

    return f"""
    <div style="color:var(--text); line-height:1.8; font-size:15px;">
        <p style="color:var(--text); margin-bottom:18px;">
            系統會根據 FaceMesh Landmark 計算臉部比例，但這些幾何特徵只作為輔助說明，
            不會取代 ResNet34 的主要分類結果。
        </p>

        {geometry_reliability_bar("臉部長度比例", f"{face_len:.3f}", face_len_note, face_len_pct, face_len_desc)}
        {geometry_reliability_bar("上臉寬度比例", f"{upper_ratio:.3f}", upper_note, upper_pct, upper_desc)}
        {geometry_reliability_bar("下顎寬度比例", f"{jaw_ratio:.3f}", jaw_note, jaw_pct, jaw_desc)}
        {geometry_reliability_bar("下顎線條角度", f"{jaw_angle:.1f}°", jaw_angle_note, jaw_angle_pct, jaw_angle_desc)}

        <p style="color:var(--muted); font-size:13px; line-height:1.75; margin-top:18px;">
            此區塊主要用來提高系統可解釋性。臉型判斷仍以模型分類結果、機率分布與 Top1 / Top2 差距為主。
        </p>
    </div>
    """

def geometry_reliability_bar(label, value, reliability, percent, desc):
    """
    產生幾何輔助分析用的單一特徵卡。

    label：特徵名稱，例如臉部長度比例
    value：實際數值，例如 1.292
    reliability：可信度文字，例如 可參考 / 低可信
    percent：橫條長度
    desc：這個特徵的白話說明
    """
    return f"""
    <div style="margin-bottom:20px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:7px;">
            <span style="color:var(--text); font-weight:900;">▸ {label}</span>
            <span style="color:var(--primary-active); font-weight:900;">{value}｜{reliability}</span>
        </div>

        <div style="height:10px; background:#E8DDD8; border-radius:999px; overflow:hidden;">
            <div style="width:{percent}%; height:100%; background:var(--primary); border-radius:999px;"></div>
        </div>

        <p style="color:var(--muted); font-size:13px; line-height:1.7; margin:7px 0 0 0;">
            {desc}
        </p>
    </div>
    """


# 分析總覽整個 Tab
def build_summary_tabs(tips, prob_bars_html, composition_text, feature_html, confidence_html):
    """
    產生右下角「分析總覽」裡面的三個小分頁。

    這裡只負責把已經做好的 HTML 區塊組合起來：
    - 髮型推薦
    - 臉型分析
    - 幾何輔助
    """
    return f"""
    <div class="summary-card">

        <div class="result-tabs">
            <input type="radio" id="result-tab_recommend" name="result_tabs" checked>
            <label for="result-tab_recommend">髮型推薦</label>

            <input type="radio" id="result-tab_face" name="result_tabs">
            <label for="result-tab_face">臉型分析</label>

            <input type="radio" id="result-tab_feature" name="result_tabs">
            <label for="result-tab_feature">幾何輔助</label>

            <input type="radio" id="result-tab_confidence" name="result_tabs">
            <label for="result-tab_confidence">分析可信度</label>

            <div class="result-tab-content recommend-content">
                <div class="summary-item">
                    <h4>智慧推薦方向</h4>
                    <div style="color:var(--text); opacity:1;">
                        {tips}
                    </div>
                </div>
            </div>

            <div class="result-tab-content face-content">
                <div class="summary-item">
                    <h4>臉型機率分布</h4>
                    {prob_bars_html}
                </div>

                <div class="summary-item">
                    <h4>臉型組成分析</h4>
                    {composition_text}
                </div>
            </div>

            <div class="result-tab-content feature-content">
                <div class="summary-item">
                    <h4>幾何輔助分析</h4>
                    {feature_html}
                </div>
            </div>

            <div class="result-tab-content confidence-content">
                <div class="summary-item">
                    <h4>分析可信度</h4>
                    {confidence_html}
                </div>
            </div>
        </div>

    </div>
    """
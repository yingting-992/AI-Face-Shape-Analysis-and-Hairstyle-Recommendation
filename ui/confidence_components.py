# confidence_components.py

# 計算出來的可信度報告，畫面上顯示 HTML
def build_confidence_html(report):
    """
    把 confidence report 轉成畫面 HTML。
    """

    messages_html = "".join(
        [f"<li>{msg}</li>" for msg in report["messages"]]
    )

    landmark_text = "Detected" if report["landmark_detected"] else "Not Detected"
    geo_text = "Used" if report["geometry_used"] else "Not Used"

    return f"""
    <div class="confidence-card">
        <h3>Analysis Confidence</h3>

        <div class="confidence-score">
            <strong>{report["stars"]}</strong>
            <span>{report["level"]}｜{report["label"]}</span>
        </div>

        <div class="confidence-grid">
            <div class="confidence-item">
                <b>Model Confidence</b>
                <span>{report["model_confidence"]:.2f}</span>
            </div>

            <div class="confidence-item">
                <b>Top1 Gap</b>
                <span>{report["top_gap"]:.2f}</span>
            </div>

            <div class="confidence-item">
                <b>Landmark</b>
                <span>{landmark_text}</span>
            </div>

            <div class="confidence-item">
                <b>Geometry</b>
                <span>{geo_text}</span>
            </div>
        </div>

        <div class="confidence-assessment">
            <h4>System Assessment</h4>
            <ul>
                {messages_html}
            </ul>
        </div>
    </div>
    """
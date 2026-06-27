# 推薦知識庫頁面

# ui/knowledge_base.py

def build_kb_statistics(rules: dict):
    """
    產生推薦知識庫上方的統計卡片。

    這裡只是統計 JSON 裡有多少臉型、
    有多少避免規則、有多少來源。
    """
    face_shapes = rules

    total_avoid = sum(
        len(v.get("avoid", [])) for v in face_shapes.values()
    )

    total_sources = 0
    for data in face_shapes.values():
        for item in data.get("recommend", []):
            total_sources += len(item.get("sources", []))
        for item in data.get("avoid", []):
            total_sources += len(item.get("sources", []))

    return f"""
<div class="kb-stats-grid">
  <div class="kb-stat-card"><b>{len(face_shapes)}</b><span>Face Shapes</span></div>
  <div class="kb-stat-card"><b>{total_avoid}</b><span>Avoid Hairstyles</span></div>
  <div class="kb-stat-card"><b>{total_sources}</b><span>Knowledge Sources</span></div>
</div>
"""


def build_knowledge_base_html(rules: dict):
    """
    產生每個臉型的推薦知識卡片。

    這頁的目的不是讓使用者操作，
    而是展示：推薦不是亂配，是有規則依據的。
    """
    cards = ""

    for shape, data in rules.items():
        principles = "".join(
            [f"<li>{item}</li>" for item in data.get("design_principles", [])]
        )

        recommended = "".join(
            [f"<span class='kb-chip'>{item.get('name', '')}</span>" for item in data.get("recommend", [])]
        )

        avoid = "".join(
            [f"<span class='kb-chip avoid'>{item.get('name', '')}</span>" for item in data.get("avoid", [])]
        )

        cards += f"""
        <div class="kb-shape-card">
            <h3>{shape}</h3>

            <div class="kb-card-section">
                <h4>設計原則</h4>
                <ul>{principles}</ul>
            </div>

            <div class="kb-card-section">
                <h4>推薦方向</h4>
                <div class="kb-chip-wrap">{recommended}</div>
            </div>

            <div class="kb-card-section">
                <h4>避免方向</h4>
                <div class="kb-chip-wrap">{avoid}</div>
            </div>
        </div>
        """

    return f"""
<h2>Recommendation Knowledge Base</h2>
<p class="kb-desc">
本頁整理系統的髮型推薦知識規則，用來說明推薦邏輯的依據。
</p>

<div class="kb-card-grid">
    {cards}
</div>
"""


def build_sources_html(rules: dict):
    """
    產生 Knowledge Sources 區塊。

    這裡會把每個臉型底下的推薦/避免規則來源列出來，
    讓人知道知識庫不是憑空寫的。
    """
    html = """
<h2>Knowledge Sources</h2>
<p class="kb-desc">
以下列出各臉型規則所參考的知識來源，用來輔助說明推薦與避免方向的依據。
</p>
<div class="kb-source-grid">
"""

    for shape, data in rules.items():
        html += f"""
<div class="kb-source-card">
  <h3>{shape}</h3>
"""

        for group_name, label in [("recommend", "推薦"), ("avoid", "避免")]:
            items = data.get(group_name, [])

            if not items:
                continue

            html += f"""
  <div class="kb-source-group">
    <h4>{label}方向</h4>
"""

            for item in items:
                item_name = item.get("name", "")
                sources = item.get("sources", [])
                source_text = "、".join(sources) if sources else "未列出來源"

                html += f"""
    <div class="kb-source-item">
      <b>{item_name}</b>
      <span>{source_text}</span>
    </div>
"""

            html += """
  </div>
"""

        html += """
</div>
"""

    html += """
</div>
"""
    return html
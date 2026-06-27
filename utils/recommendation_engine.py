import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RULE_PATH = BASE_DIR / "knowledge" / "hairstyle_rules.json"
REF_PATH = BASE_DIR / "knowledge" / "references.json"

def load_hairstyle_rules():
    with open(RULE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# 格式化推薦文字
def recommend_hairstyles(face_shape: str, top_k: int = 6):
    rules = load_hairstyle_rules()

    if face_shape not in rules:
        return {
            "recommended": [],
            "avoid": [],
            "design_principles": [],
            "sources": [],
            "summary": f"目前尚未建立 {face_shape} 臉型的推薦規則。"
        }

    data = rules[face_shape]
    recommended = data.get("recommended", [])[:top_k]
    avoid = data.get("avoid", [])

    source_names = []
    for item in recommended + avoid:
        source_names.extend(item.get("sources", []))

    return {
        "recommended": recommended,
        "avoid": avoid,
        "design_principles": data.get("design_principles", []),
        "sources": source_names,
        "summary": f"根據系統判斷的 {face_shape} 臉型，以下提供適合的髮型方向與避免建議。"
    }

# 格式化推薦文字
def _chips(items, bg="rgba(120,90,75,0.12)", color="var(--text)"):
    if not items:
        return ""
    return "".join(
        f"<span style='display:inline-block; padding:4px 9px; margin:3px 5px 3px 0; "
        f"border-radius:999px; background:{bg}; color:{color}; font-size:12px; font-weight:800;'>"
        f"{item}</span>"
        for item in items
    )


def format_recommendation_text(result: dict):
    lines = []

    if result.get("summary"):
        lines.append(
            f"<p style='color:var(--text); line-height:1.8; margin:0 0 14px 0;'>"
            f"{result['summary']}"
            f"</p>"
        )

    if result.get("design_principles"):
        lines.append("<div style='margin:14px 0 20px 0;'>")
        lines.append("<div style='color:var(--text); font-size:15px; font-weight:900; margin-bottom:8px;'>推薦原則</div>")
        lines.append(_chips(result["design_principles"], bg="#E8DDD8", color="#3A2822"))
        lines.append("</div>")

    lines.append("<div style='color:var(--text); font-size:15px; font-weight:900; margin:16px 0 10px 0;'>推薦髮型</div>")

    for i, item in enumerate(result.get("recommended", []), start=1):
        name = item.get("name", "未命名髮型")
        reason = item.get("reason", "")
        basis = item.get("basis", [])
        tags = item.get("tags", [])

        lines.append(
            f"<div style='padding:14px 16px; margin-bottom:12px; border:1px solid var(--line); "
            f"border-radius:18px; background:rgba(255,255,255,0.35);'>"
            f"<div style='color:var(--primary-active); font-size:15px; font-weight:900; margin-bottom:8px;'>"
            f"{i}. {name}</div>"
            f"<div style='color:var(--text); line-height:1.75; font-size:14px; margin-bottom:8px;'>"
            f"原因：{reason}</div>"
            f"<div style='margin-bottom:6px;'><span style='color:var(--text); font-weight:900; font-size:13px;'>依據：</span>"
            f"{_chips(basis, bg='rgba(120,90,75,0.12)', color='var(--text)')}</div>"
            f"<div><span style='color:var(--text); font-weight:900; font-size:13px;'>標籤：</span>"
            f"{_chips(tags, bg='rgba(167,123,104,0.14)', color='var(--text)')}</div>"
            f"</div>"
        )

    lines.append("<div style='color:var(--text); font-size:15px; font-weight:900; margin:18px 0 10px 0;'>不建議髮型</div>")

    for item in result.get("avoid", []):
        name = item.get("name", "未命名髮型")
        reason = item.get("reason", "")
        basis = item.get("basis", [])

        lines.append(
            f"<div style='padding:13px 16px; margin-bottom:10px; border:1px solid var(--line); "
            f"border-radius:18px; background:rgba(255,255,255,0.25);'>"
            f"<div style='color:var(--primary-active); font-size:14px; font-weight:900; margin-bottom:8px;'>"
            f"• {name}</div>"
            f"<div style='color:var(--text); line-height:1.75; font-size:14px; margin-bottom:8px;'>"
            f"原因：{reason}</div>"
            f"<div><span style='color:var(--text); font-weight:900; font-size:13px;'>依據：</span>"
            f"{_chips(basis, bg='rgba(120,90,75,0.12)', color='var(--text)')}</div>"
            f"</div>"
        )

    sources = sorted(set(result.get("sources", [])))
    if sources:
        lines.append(
            f"<div style='margin-top:18px; padding-top:14px; border-top:1px dashed var(--line);'>"
            f"<div style='color:var(--text); font-size:15px; font-weight:900; margin-bottom:8px;'>知識庫來源</div>"
            f"{_chips(sources, bg='#E8DDD8', color='#3A2822')}"
            f"<p style='color:var(--muted); font-size:12px; line-height:1.7; margin-top:8px;'>"
            f"推薦規則整理自公開髮型與沙龍建議資料，僅作為造型參考。</p>"
            f"</div>"
        )

    return "\n".join(lines)
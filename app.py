# Gradio 網頁入口（上傳圖→推論→畫廊→建議）
from pathlib import Path
import random
import os

from dotenv import load_dotenv
import gradio as gr
from PIL import Image

from core.infer_core import HAIR_RULES, load_model, infer_once


# ===================== 可調整常數 =====================
load_dotenv()

CKPT_PATH = Path(os.getenv("CKPT_PATH", "checkpoints/resnet34_faceshape_best_grid_ES_6.pth"))
HAAR_PATH = Path(os.getenv("HAAR_PATH", "models/haarcascade_frontalface_default.xml"))
ASSETS_DIR = Path(os.getenv("ASSETS_DIR", "assets/hairstyles"))

LOW_CONF = float(os.getenv("LOW_CONF", 0.55))
GALLERY_K = int(os.getenv("GALLERY_K", 6))

APP_CSS = """
/* === 共用：字色與背景 === */
body { background-color: #FAFAFA; }

/* 上傳框：顯示為淡色卡片 */
#upload_box {
  border: 2px dashed #E6CFC6;
  background: #FFF9F5;
  border-radius: 14px;
}

/* 結果標題 */
#result_title {
  color: #444;
  font-weight: 600;
}

/* 摘要卡片 */
#summary_box textarea {
  background: #FFFFFF;
  border: 1px solid #EEE;
  border-radius: 14px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* 圖庫容器 */
#gallery_box {
  background: #F7F9FA;
  border-radius: 12px;
  padding: 8px;
}
"""


# ===================== 啟動時載入一次 =====================
_model, CLASS_NAMES, IMG_SIZE, _tf = load_model(CKPT_PATH)


# ===================== 提供給 Gradio 的包裝函式 =====================
def ui_infer(image: Image.Image, use_haar: bool, low_conf: float, topk_gallery: int):
    if image is None:
        return None, "## ⚠️ 請先上傳照片", [], "請先上傳照片再開始分析。"

    out = infer_once(
        image_pil=image,
        model=_model,
        class_names=CLASS_NAMES,
        tf=_tf,
        use_haar=use_haar,
        haar_path=HAAR_PATH,
        low_conf=low_conf,
    )

    if "error" in out:
        return None, f"❌ {out['error']}", [], f"⚠️ {out['error']}"

    vis_pil = out["vis_pil"]
    top1, p1, top2, p2 = out["top1"], out["p1"], out["top2"], out["p2"]
    used_geo, geo_msg, tips = out["used_geo"], out["geo_msg"], out["tips"]

    title = f"## 臉型判定結果：{top1}（{p1:.2f}）"
    if used_geo:
        title += "  ⚙️ *(已啟用幾何補票)*"
    if geo_msg:
        title += f"\n\n> {geo_msg}"

    folder = ASSETS_DIR / top1
    gallery = []
    if folder.exists():
        files = [p for p in folder.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
        random.shuffle(files)
        gallery = [str(p) for p in files[: int(topk_gallery)]]

    rule = HAIR_RULES.get(top1, {})
    do = "、".join(rule.get("do", []))
    avoid = "、".join(rule.get("avoid", []))
    note = rule.get("notes", "")
    summary_md = f"""
    ## 💡 髮型建議摘要
    ---
    ### ✅ 推薦：{do if do else '無特別建議'}

    ### ❌ 避免：{avoid if avoid else '無明顯禁忌'}

    ### 📝 備註：{note if note else '—'}

    ### ⚠️ 使用說明
    此建議主要依據臉型分類結果與髮型規則產生，實際髮型仍需考量髮量、髮質、瀏海習慣、個人風格與設計師判斷。
    """

    return vis_pil, title, gallery, summary_md


# ===================== Gradio 介面 =====================
with gr.Blocks(title="臉型分析與髮型建議") as demo:
    gr.Markdown(
        """
        <h1 style="text-align:center; color:#000000;">💇‍♀️ 臉型分析與髮型建議系統</h1>
        <p style="text-align:center; font-size:16px; color:#555;">
        上傳正臉照片（請保持光線均勻），系統將自動辨識臉型並提供個人化髮型建議與參考圖片。
        </p>
        <p style="text-align:center; font-size:14px; color:#888;">
        ※ 本系統目前以公開 Face Shape Dataset 訓練，資料來源以歐美臉孔為主，亞洲臉孔泛化能力仍需後續驗證；分析結果僅供髮型參考，不代表絕對判定。
        </p>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="📸 上傳照片", elem_id="upload_box")
            use_haar = gr.Checkbox(value=False, label="使用 Haar 備援模式（無 landmark，只做臉框裁切）")
            low_conf = gr.Slider(
                0.3, 0.9, value=LOW_CONF, step=0.01,
                label="低信心門檻（啟用幾何補票）",
                info="當模型信心值低於此門檻時，自動啟動幾何補判機制",
            )
            topk_gallery = gr.Slider(3, 9, value=GALLERY_K, step=1, label="參考圖片數量")
            btn = gr.Button("🚀 開始分析", variant="primary")

        with gr.Column(scale=2):
            title_out = gr.Markdown("## 🔍 分析結果", elem_id="result_title")
            vis_out = gr.Image(label="關鍵點/對齊預覽", show_label=False, elem_id="vis_out")
            gallery = gr.Gallery(
                label="✨ 適合髮型參考",
                columns=3,
                height=300,
                preview=True,
                elem_id="gallery_box",
            )
            summary = gr.Markdown("等待分析結果...", elem_id="summary_box")

    btn.click(
        fn=ui_infer,
        inputs=[img_in, use_haar, low_conf, topk_gallery],
        outputs=[vis_out, title_out, gallery, summary],
    )


# ===================== 啟動 Gradio 服務 =====================
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=APP_CSS,
        inbrowser=True,
    )

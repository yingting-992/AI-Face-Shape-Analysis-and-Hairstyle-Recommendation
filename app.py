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
:root {
  --bg: #F8F1EC;
  --card: #FFFFFF;
  --card-2: #FFF6F0;
  --text: #2E211C;
  --title: #271A16;
  --muted: #6E554B;
  --line: rgba(136, 91, 72, 0.24);
  --primary: #A77B68;
  --primary-hover: #8D6455;
  --primary-active: #6F4A40;
  --label-bg: #5A3B31;
  --label-bg-soft: #9A6852;
  --shadow: 0 18px 42px rgba(98, 66, 54, 0.12);
  --soft-shadow: 0 10px 26px rgba(98, 66, 54, 0.09);
}

html.dark-mode {
  --bg: #12100F;
  --card: #201A18;
  --card-2: #29211E;
  --text: #F2E6DF;
  --title: #FFF7F2;
  --muted: #CBB8AE;
  --line: rgba(232, 210, 199, 0.20);
  --primary: #C7A092;
  --primary-hover: #D1A99A;
  --primary-active: #9A7163;
  --shadow: 0 18px 42px rgba(0, 0, 0, 0.36);
  --soft-shadow: 0 10px 26px rgba(0, 0, 0, 0.25);
}

html,
body {
  margin: 0 !important;
  padding: 0 !important;
  width: 100% !important;
  min-height: 100% !important;
  background: var(--bg) !important;
  overflow-x: hidden !important;
}

body,
.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(219, 190, 176, 0.24), transparent 30%),
    radial-gradient(circle at top right, rgba(239, 218, 207, 0.34), transparent 34%),
    linear-gradient(180deg, var(--bg) 0%, #FBF6F2 58%, var(--bg) 100%) !important;
  color: var(--text) !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans TC", "Microsoft JhengHei", sans-serif !important;
}

html.dark-mode body,
html.dark-mode .gradio-container {
  background:
    radial-gradient(circle at top left, rgba(84, 62, 54, 0.24), transparent 30%),
    radial-gradient(circle at top right, rgba(70, 54, 49, 0.28), transparent 34%),
    var(--bg) !important;
}

.gradio-container {
  max-width: none !important;
  width: 100vw !important;
  margin: 0 !important;
  padding: 28px 44px 34px 44px !important;
  box-sizing: border-box !important;
}

/* ===== 主題切換器 ===== */
#theme_toggle {
  position: fixed;
  top: 18px;
  right: 22px;
  z-index: 9999;
  display: flex;
  align-items: center;
  gap: 8px;
  height: 34px;
  padding: 0 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid var(--line);
  box-shadow: var(--soft-shadow);
  backdrop-filter: blur(12px);
  color: #4A332B !important;
  font-size: 12px;
  font-weight: 900;
  white-space: nowrap;
  overflow: hidden;
}

html.dark-mode #theme_toggle {
  background: rgba(32, 26, 24, 0.86);
  color: #F2E6DF !important;
}

#theme_toggle span {
  color: inherit !important;
  opacity: 1 !important;
  font-weight: 900 !important;
}

#theme_toggle input {
  display: none;
}

#theme_toggle label {
  width: 42px;
  height: 22px;
  display: block;
  margin: 0 !important;
  padding: 0 !important;
  line-height: 0 !important;
  overflow: hidden;
}

#theme_toggle .switch {
  width: 42px;
  height: 22px;
  display: block;
  border-radius: 999px;
  background: #D6C1B8;
  position: relative;
  cursor: pointer;
  transition: 0.22s ease;
  overflow: hidden;
}

#theme_toggle .switch::after {
  content: "";
  position: absolute;
  width: 17px;
  height: 17px;
  top: 2.5px;
  left: 3px;
  border-radius: 50%;
  background: #FFFFFF;
  box-shadow: 0 2px 7px rgba(0,0,0,0.22);
  transition: 0.22s ease;
}

#theme_toggle input:checked + .switch {
  background: #765D54;
}

#theme_toggle input:checked + .switch::after {
  left: 22px;
}

/* ===== 標題區 ===== */
#hero {
  margin: 10px auto 28px auto;
  padding: 30px 20px 24px 20px;
  text-align: center;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

#hero .eyebrow {
  display: inline-block;
  margin-bottom: 12px;
  color: var(--muted) !important;
  font-size: 12px;
  letter-spacing: 0.16em;
  font-weight: 850;
}

#hero h1 {
  margin: 0 0 13px 0;
  font-size: 38px;
  letter-spacing: 0.05em;
  color: var(--title) !important;
  font-weight: 900;
}

#hero .subtitle {
  margin: 0 auto;
  max-width: 980px;
  font-size: 16px;
  line-height: 1.9;
  color: var(--muted) !important;
  font-weight: 560;
}

#hero .notice {
  margin: 16px auto 0 auto;
  max-width: 1050px;
  font-size: 13px;
  line-height: 1.8;
  color: var(--muted) !important;
  font-weight: 500;
}

/* ===== 主版面卡片 ===== */
#main_row {
  gap: 24px !important;
}

#upload_panel,
#result_panel,
#recommend_panel {
  background: var(--card) !important;
  border-radius: 28px !important;
  border: 1px solid var(--line) !important;
  box-shadow: var(--shadow) !important;
  padding: 24px !important;
}

.section-title {
  font-size: 21px;
  font-weight: 900;
  color: var(--title) !important;
  margin: 0 0 8px 0;
  letter-spacing: 0.04em;
}

.section-desc {
  font-size: 13px;
  color: var(--muted) !important;
  margin: 0 0 18px 0;
  line-height: 1.8;
  font-weight: 560;
}

/* ===== 上傳與預覽框 ===== */
#upload_box,
#vis_out {
  background: var(--card) !important;
  border-radius: 24px !important;
  border: 1px solid var(--line) !important;
  box-shadow: var(--soft-shadow) !important;
  overflow: hidden !important;
}

#vis_out {
  padding: 8px !important;
}

#vis_out img {
  max-height: 540px !important;
  object-fit: contain !important;
}

/* ===== Gradio 圖片元件左上角 label：咖啡底白字 ===== */
/* 只改左上角小標籤，不要改整個上傳區 */
#upload_box .label-wrap,
#vis_out .label-wrap,
#gallery_box .label-wrap {
  background: var(--label-bg) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 8px !important;
}

#upload_box .label-wrap *,
#vis_out .label-wrap *,
#gallery_box .label-wrap * {
  color: #FFFFFF !important;
  fill: #FFFFFF !important;
  stroke: #FFFFFF !important;
  font-weight: 850 !important;
}

/* 拿掉小標籤前面的圖示 */
#upload_box .label-wrap svg,
#vis_out .label-wrap svg,
#gallery_box .label-wrap svg {
  display: none !important;
}
/* ===== 控制區 ===== */
#control_card {
  margin-top: 16px;
  padding: 18px !important;
  border-radius: 24px !important;
  background: #F8F0EA !important;
  border: 1px solid rgba(136, 91, 72, 0.22) !important;
  box-shadow: 0 10px 26px rgba(98, 66, 54, 0.08) !important;
}

html.dark-mode #control_card {
  background: #211A17 !important;
  border: 1px solid rgba(232, 210, 199, 0.18) !important;
}

#control_card label,
#control_card span,
#control_card p,
#control_card small {
  color: var(--text) !important;
  opacity: 1 !important;
}

#control_card > div,
#control_card .block,
#control_card .form,
#control_card .wrap,
#control_card .gr-group,
#control_card .gr-box,
#control_card .gr-panel,
#control_card .contain,
#control_card .container,
#control_card fieldset {
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
}

/* Slider 標籤：低信心門檻、髮型參考圖片數量 */
#control_card span[data-testid="block-info"] {
  background: var(--label-bg-soft) !important;
  color: #FFFFFF !important;
  border-radius: 8px !important;
  padding: 6px 10px !important;
  display: inline-block !important;
  font-weight: 850 !important;
}

#control_card span[data-testid="block-info"] * {
  color: #FFFFFF !important;
}

#control_card .info-text {
  color: #8A746B !important;
}

html.dark-mode #control_card .info-text {
  color: #CBB8AE !important;
}

/* Checkbox */
#control_card input[type="checkbox"] {
  appearance: none !important;
  -webkit-appearance: none !important;
  width: 15px !important;
  height: 15px !important;
  border: 2px solid #5A3B31 !important;
  border-radius: 4px !important;
  background: #FFFFFF !important;
  opacity: 1 !important;
  margin-right: 8px !important;
  vertical-align: middle !important;
}

#control_card input[type="checkbox"]:checked {
  background: #5A3B31 !important;
  border-color: #5A3B31 !important;
}

#control_card input[type="checkbox"]:checked::after {
  content: "✓";
  color: #FFFFFF !important;
  font-size: 11px !important;
  font-weight: 900 !important;
  position: relative;
  left: 2px;
  top: -5px;
}

/* Slider */
#control_card input[type="range"] {
  display: block !important;
  visibility: visible !important;
  opacity: 1 !important;
  width: 100% !important;
  height: 22px !important;
  accent-color: #6F4A40 !important;
}

#control_card input[type="range"]::-webkit-slider-runnable-track {
  background: #D8C7BE !important;
  height: 6px !important;
  border-radius: 999px !important;
}

#control_card input[type="range"]::-webkit-slider-thumb {
  background: #6F4A40 !important;
  border: 2px solid #FFFFFF !important;
  border-radius: 999px !important;
}

#control_card input[type="range"]::-moz-range-track {
  background: #D8C7BE !important;
  height: 6px !important;
  border-radius: 999px !important;
}

#control_card input[type="range"]::-moz-range-progress {
  background: #6F4A40 !important;
  height: 6px !important;
  border-radius: 999px !important;
}

#control_card input[type="range"]::-moz-range-thumb {
  background: #6F4A40 !important;
  border: 2px solid #FFFFFF !important;
  border-radius: 999px !important;
}

#control_card input[type="number"],
#control_card input[type="text"] {
  background: #E6D4CA !important;
  color: #3A2822 !important;
  border: 1px solid rgba(111, 74, 64, 0.22) !important;
  border-radius: 8px !important;
}

html.dark-mode #control_card input[type="number"],
html.dark-mode #control_card input[type="text"] {
  background: #332824 !important;
  color: #F2E6DF !important;
  border-color: rgba(232, 210, 199, 0.18) !important;
}

#control_card button {
  background: #E6D4CA !important;
  color: #3A2822 !important;
  border: 1px solid rgba(111, 74, 64, 0.22) !important;
  border-radius: 8px !important;
}

/* ===== 開始分析按鈕 ===== */
#analyze_btn {
  border-radius: 999px !important;
  min-height: 50px !important;
  font-weight: 900 !important;
  letter-spacing: 0.12em !important;
  background: linear-gradient(135deg, var(--primary) 0%, var(--primary-hover) 100%) !important;
  color: #ffffff !important;
  border: none !important;
  box-shadow: 0 8px 22px rgba(160, 116, 96, 0.32) !important;
  transition: transform 0.16s ease, box-shadow 0.16s ease, filter 0.16s ease !important;
}

#analyze_btn:hover {
  transform: translateY(-2px);
  filter: brightness(1.03);
  box-shadow: 0 12px 28px rgba(160, 116, 96, 0.48) !important;
}

#analyze_btn:active {
  transform: translateY(1px) scale(0.985);
  filter: brightness(0.88);
  background: linear-gradient(135deg, var(--primary-active) 0%, #67473E 100%) !important;
}

/* ===== 分析結果文字卡片 ===== */
#result_title {
  padding: 18px 20px !important;
  border-radius: 22px !important;
  background: linear-gradient(135deg, var(--card), var(--card-2)) !important;
  border: 1px solid var(--line) !important;
  color: var(--text) !important;
  opacity: 1 !important;
  box-shadow: var(--soft-shadow) !important;
  margin-bottom: 18px !important;
}

#result_title,
#result_title * {
  color: var(--text) !important;
  opacity: 1 !important;
  text-shadow: none !important;
}

/* ===== 髮型推薦 ===== */
#summary_box {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
}

.summary-card {
  background: linear-gradient(to bottom right, var(--card), var(--card-2));
  border-left: 5px solid var(--primary);
  padding: 22px 24px;
  border-radius: 0 22px 22px 0;
  box-shadow: var(--soft-shadow);
  color: var(--text);
}

.summary-item {
  margin-bottom: 18px;
  padding-bottom: 18px;
  border-bottom: 1px dashed var(--line);
}

.summary-item:last-child {
  border-bottom: none;
}

.summary-item h3,
.summary-item h4 {
  margin-top: 0;
  letter-spacing: 0.03em;
  color: var(--text) !important;
}

.summary-item h4 {
  font-weight: 900 !important;
  font-size: 16px !important;
}

.summary-item p {
  color: var(--text) !important;
}

#gallery_box {
  margin-top: 18px !important;
  background: var(--card) !important;
  border-radius: 24px !important;
  padding: 14px !important;
  border: 1px solid var(--line) !important;
  box-shadow: var(--soft-shadow) !important;
}

/* ===== Footer ===== */
#footer {
  text-align: center;
  color: var(--muted) !important;
  font-size: 12px;
  margin: 26px 0 10px 0;
}

/* ===== 手機版 ===== */
@media (max-width: 900px) {
  .gradio-container {
    width: 100vw !important;
    padding: 14px !important;
  }

  #hero {
    padding: 24px 12px 18px 12px;
  }

  #hero h1 {
    font-size: 26px;
  }

  #theme_toggle {
    top: 10px;
    right: 10px;
    transform: scale(0.92);
  }
}
"""


# ===================== 啟動時載入一次 =====================
_model, CLASS_NAMES, IMG_SIZE, _tf = load_model(CKPT_PATH)


# ===================== 提供給 Gradio 的包裝函式 =====================
def ui_infer(image: Image.Image, use_haar: bool, low_conf: float, topk_gallery: int):
    if image is None:
        return None, "## 請先上傳照片", [], "請先上傳正臉照片再開始分析。"

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
        error_md = f"""
<div class="summary-card">
    <div class="summary-item" style="margin-bottom:0; padding-bottom:0;">
        <h3 style="color:#B3261E; margin-top:0;">無法完成分析</h3>
        <p style="line-height:1.8;">{out['error']}</p>
    </div>
</div>
"""
        return None, f"## 無法完成分析\n\n{out['error']}", [], error_md

    vis_pil = out["vis_pil"]
    top1, p1, top2, p2 = out["top1"], out["p1"], out["top2"], out["p2"]
    used_geo, geo_msg = out["used_geo"], out["geo_msg"]

    title = f"""
## 臉型判定結果：{top1}

**模型信心值：{p1:.2f}**  
第二可能結果：{top2}（{p2:.2f}）
"""
    if used_geo:
        title += "\n\n已啟用低信心幾何補判機制。"
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
<div class="summary-card">
    <div class="summary-item">
        <h3 style="color: var(--text); margin-top:0;">
            臉型判定結果：
            <strong style="color: var(--primary-active); font-size: 24px;">{top1}</strong>
        </h3>
        <p style="color: var(--muted); font-size: 14px; margin-bottom:0;">
            模型信心值：{p1:.2f}
            {'（已啟動幾何特徵輔助判定）' if used_geo else ''}
        </p>
    </div>

    <div class="summary-item">
        <h4>智慧推薦方向</h4>
        <p style="line-height: 1.75; font-size: 15px;">
            {do if do else '根據您的臉型，暫無特定推薦，維持自然線條即可。'}
        </p>
    </div>

    <div class="summary-item">
        <h4>建議避免地雷</h4>
        <p style="line-height: 1.75; font-size: 15px;">
            {avoid if avoid else '無明顯禁忌髮型。'}
        </p>
    </div>

    <div class="summary-item">
        <h4>專業設計備註</h4>
        <p style="color: var(--muted); line-height: 1.75; font-size: 14px;">
            {note if note else '實際髮型仍需考量您的髮量、髮質與日常整理習慣。'}
        </p>
    </div>

    <div class="summary-item" style="margin-bottom:0; padding-bottom:0;">
        <h4>使用說明</h4>
        <p style="color: var(--muted); line-height: 1.75; font-size: 13px; margin-bottom:0;">
            此建議主要依據臉型分類結果與髮型規則產生，實際髮型仍需考量髮量、髮質、瀏海習慣、個人風格與設計師判斷。
        </p>
    </div>
</div>
"""

    return vis_pil, title, gallery, summary_md


# ===================== Gradio 介面 =====================
with gr.Blocks(
    title="AI 臉型分析與髮型推薦系統",
    theme=gr.themes.Soft(
        primary_hue="stone",
        secondary_hue="neutral",
        neutral_hue="slate",
    ),
    css=APP_CSS,
) as demo:
    gr.HTML(
        """
        <div id="theme_toggle">
          <span>淺色</span>
          <label>
            <input type="checkbox" onchange="document.documentElement.classList.toggle('dark-mode', this.checked)">
            <span class="switch"></span>
          </label>
          <span>深色</span>
        </div>

        <div id="hero">
          <div class="eyebrow">AI BEAUTY CONSULTANT</div>
          <h1>AI 臉型分析與智慧髮型推薦系統</h1>
          <p class="subtitle">
            上傳正臉照片後，系統會進行人臉偵測、Landmark 分析、臉型辨識與髮型推薦，
            並提供推薦理由與參考圖片。
          </p>
          <div class="notice">
            本系統目前以公開 Face Shape Dataset 訓練，資料來源以歐美臉孔為主；
            亞洲臉孔泛化能力仍需後續驗證。分析結果僅供髮型參考，不代表絕對判定。
          </div>
        </div>
        """
    )

    with gr.Row(equal_height=False, elem_id="main_row"):
        with gr.Column(scale=5, min_width=430, elem_id="upload_panel"):
            gr.HTML('<div class="section-title">上傳照片</div>')
            gr.HTML('<div class="section-desc">建議使用光線均勻、臉部完整、角度接近正面的照片。</div>')

            img_in = gr.Image(type="pil", label="照片上傳", elem_id="upload_box")

            with gr.Column(elem_id="control_card"):
                use_haar = gr.Checkbox(
                    value=False,
                    label="使用 Haar 備援模式（無 landmark，只做臉框裁切）",
                )

                low_conf = gr.Slider(
                    0.3, 0.9, value=LOW_CONF, step=0.01,
                    label="低信心門檻",
                    info="當模型信心值低於此門檻時，自動啟動幾何補判機制",
                )

                topk_gallery = gr.Slider(
                    3, 9, value=GALLERY_K, step=1,
                    label="髮型參考圖片數量",
                )

            btn = gr.Button("開始分析", variant="primary", elem_id="analyze_btn")

        with gr.Column(scale=12, min_width=780):
            with gr.Column(elem_id="result_panel"):
                gr.HTML('<div class="section-title">分析結果</div>')
                title_out = gr.Markdown("## 尚未分析\n\n請先上傳照片並點選開始分析。", elem_id="result_title")
                vis_out = gr.Image(label="Landmark / 對齊預覽", show_label=True, elem_id="vis_out")

            gr.HTML("<br>")

            with gr.Column(elem_id="recommend_panel"):
                gr.HTML('<div class="section-title">髮型推薦</div>')
                summary = gr.HTML("等待分析結果...", elem_id="summary_box")
                gallery = gr.Gallery(
                    label="適合髮型參考",
                    columns=3,
                    height=360,
                    preview=True,
                    elem_id="gallery_box",
                )

    gr.HTML('<div id="footer">AI Face Shape Analysis & Intelligent Hairstyle Recommendation System</div>')

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
        inbrowser=True,
    )

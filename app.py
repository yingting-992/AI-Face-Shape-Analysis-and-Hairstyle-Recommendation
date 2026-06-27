# Gradio 網頁入口（上傳圖→推論→畫廊→建議）
from pathlib import Path
import random
import os

from dotenv import load_dotenv
import gradio as gr
from PIL import Image
from core import thresholds as T
from core.infer_core import load_model, infer_once

from ui.result_components import (
    build_result_title,
    build_probability_bars,
    build_composition_text,
    build_feature_analysis_html,
    build_summary_tabs,
)

from ui.knowledge_base import (
    build_kb_statistics,
    build_knowledge_base_html,
    build_sources_html,
)


# ===================== 可調整常數 =====================
load_dotenv()

CKPT_PATH = Path(os.getenv("CKPT_PATH", "checkpoints/resnet34_faceshape_best_grid_ES_6.pth"))
HAAR_PATH = Path(os.getenv("HAAR_PATH", "models/haarcascade_frontalface_default.xml"))
ASSETS_DIR = Path(os.getenv("ASSETS_DIR", "assets/hairstyles"))

LOW_CONF = float(os.getenv("LOW_CONF", 0.55))
GALLERY_K = int(os.getenv("GALLERY_K", 6))


CSS_FILE = Path(__file__).with_name("style.css")
APP_CSS = CSS_FILE.read_text(encoding="utf-8")



# ===================== 啟動時載入一次 =====================
_model, CLASS_NAMES, IMG_SIZE, _tf = load_model(CKPT_PATH)
from utils.recommendation_engine import load_hairstyle_rules
hairstyle_rules = load_hairstyle_rules()

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

    tips = out.get("tips", "")
    prob_dict = out.get("prob_dict", {})
    geometry_info = out.get("geometry_info")

    title = build_result_title(top1, p1, top2, p2, used_geo)

    folder = ASSETS_DIR / top1
    gallery = []
    if folder.exists():
        files = [p for p in folder.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
        random.shuffle(files)
        gallery = [str(p) for p in files[: int(topk_gallery)]]

    tips = out["tips"]

    # 組合分析總覽內容
    prob_bars_html = build_probability_bars(prob_dict)              # 模型機率橫條
    composition_text = build_composition_text(top1, p1, top2, p2)   # 臉型組成說明文字
    feature_html = build_feature_analysis_html(geometry_info)       # 臉部特徵分析 HTML

    summary_md = build_summary_tabs(        # 分析總覽三個分頁
        tips=tips,
        prob_bars_html=prob_bars_html,
        composition_text=composition_text,
        feature_html=feature_html,
    )

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
        <div id="top_nav">
        <div class="nav-brand">AI Beauty Consultant</div>

        <div id="theme_toggle">
            <span>淺色</span>
            <label>
            <input type="checkbox" onchange="document.documentElement.classList.toggle('dark-mode', this.checked)">
            <span class="switch"></span>
            </label>
            <span>深色</span>
        </div>
        </div>
        """
    )

    with gr.Tab("推薦知識庫"):
        kb_stats = gr.HTML(f'<div class="kb-page">{build_kb_statistics(hairstyle_rules)}</div>')
        kb_table = gr.HTML(f'<div class="kb-page">{build_knowledge_base_html(hairstyle_rules)}</div>')
        kb_sources = gr.HTML(f'<div class="kb-page">{build_sources_html(hairstyle_rules)}</div>')

        print("hairstyle_rules keys:", hairstyle_rules.keys())

    with gr.Tab("臉型判斷"):
        gr.HTML(
            """
            <div id="hero">
            <p class="subtitle">
                上傳正臉照片後，系統會進行人臉偵測、Landmark 分析、臉型辨識與髮型推薦。
            </p>
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
                    title_out = gr.HTML(
                        "<div class='result-title-card'><h2>尚未分析</h2><p>請先上傳照片並點選開始分析。</p></div>",
                        elem_id="result_title"
                    )
                    vis_out = gr.Image(label="Landmark / 對齊預覽", show_label=True, elem_id="vis_out")

                gr.HTML("<br>")

                with gr.Column(elem_id="recommend_panel"):
                    gr.HTML('<div class="section-title">分析總覽</div>')
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

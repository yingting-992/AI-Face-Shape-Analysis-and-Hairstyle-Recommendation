v0.2.0 — 模組化與 Gradio 介面

Release Date: YYYY-MM-DD

新增

    app.py

        使用 Gradio Blocks 建立 Web 介面。

        提供圖片上傳、Haar 偵測切換、低信心門檻、參考圖數量等控制元件。

        顯示臉部關鍵點預覽、Top1/Top2 臉型與信心值、參考髮型圖庫、建議與摘要。

    圖庫支援

        依臉型分類的示範髮型圖片，顯示於 Gradio Gallery。

重構

    infer_faceshape_and_recommend.py → infer_core.py

        將原本 CLI 版本程式抽出為函式模組，方便 UI 或其他應用呼叫：

            imread_utf8、detect_landmarks、align_face、geometry_vote、load_model 等函式保留。

            原本 main() 的流程改為 ui_infer(...)，回傳結果給介面而非 print。

    程式架構分離

        infer_core.py：專注於推論、幾何補票、建議規則。

        app.py：專注於使用者介面、輸入輸出呈現。

變更

    輸入方式

        原本透過 argparse 指令列參數 → 現在改由 Gradio UI 控制元件輸入。

    輸出方式

        原本 print(...) 終端輸出 → 改為 UI 顯示（圖片、Markdown、文字方塊）。

可視化

    原本使用 cv2.imshow 視窗 → 改為 Gradio Image 顯示。

效益

    使用者體驗：由命令列程式轉為 Web 互動介面，操作更直觀。

    維護性：推論核心與 UI 分離，後續更新模型或規則時不需修改介面程式。

    可擴充性：方便未來加入更多參數控制或結果輸出形式。
# ES_6 簡介：
# 以train_faceshape_resnet34_4_1.py 當母版

import os, copy, time, random, json 
from pathlib import Path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np


torch.set_num_threads(max(1, os.cpu_count() - 1))  # [NEW] 視機器情況調整
torch.set_num_interop_threads(1)                    # 降低多算子排程開銷
# torch.use_deterministic_algorithms(True, warn_only=True)

# ============ 路徑設定 ============
# 將face-shape 目錄（裡面有 training_set / testing_set）
data_root = Path(r"C:\All small projects\AI-Face-Shape-Analysis-and-Hairstyle-Recommendation\data\face-shape")  # [NEW] 改成絕對路徑，避免不同工作目錄造成找不到資料夾

# ============ 超參數 ============
img_size   = 256       
batch_size = 16        
epochs     = 60        
lr         = 3e-4
weight_decay = 1e-4
# val_ratio  = 0.15      
patience   = 12        # 早停容忍度 [NEW] 調高一點 1012    

# ---- Device & backend ----
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    # Apple Silicon 可用 MPS；沒有就回 CPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()

# 為了可重現，關閉 benchmark 並開 deterministic
if device.type == "cuda":
    torch.backends.cudnn.benchmark = False # 讓每次結果可重現 false是
    torch.backends.cudnn.deterministic = True # 讓每次結果可重現
    # try:  # [NEW] 0309 
    #     torch.use_deterministic_algorithms(True) # 再進一步要求 PyTorch 儘量使用 deterministic 的實作，讓重跑同一組參數時更穩。
    # except Exception as e:
    #     print(f"[WARN] deterministic algorithms not fully enabled: {e}")

# device = torch.device("cpu")  # 目前裝的是 CPU 版 PyTorch

# ============ 隨機種子 ============
seed = 42
# random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
def reseed_all(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available(): # [NEW] 0309 把 CUDA 端的隨機性控制得更完整。
        torch.cuda.manual_seed(s)  # [NEW] 0309 把 CUDA 端的隨機性控制得更完整。
        torch.cuda.manual_seed_all(s) # [NEW] 0309 把 CUDA 端的隨機性控制得更完整。

def seed_worker(worker_id): # [NEW] 0309 減少資料增強與 shuffle 帶來的隨機漂移
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# ============ Transforms（函式化，方便窮舉覆蓋） ============
# 先調整大小，再資料增強
def make_transforms(img_size, rotation_deg=10):  # 參數化，窮舉時可覆蓋 rotation_deg是旋轉角度
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)), # 先調整大小，再資料增強
        transforms.RandomHorizontalFlip(p=0.5), # 隨機水平翻轉 p是機率
        transforms.RandomRotation(rotation_deg) if rotation_deg>0 else transforms.Lambda(lambda x: x),  # [NEW]
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                            std=[0.229,0.224,0.225]),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                            std=[0.229,0.224,0.225]),
    ])
    return transform_train, transform_eval  # [NEW]

# ============ 按 cfg 建立資料與模型（每個 trial 都會重建一次） ============
def build_data_and_model(cfg, trial_id):
    """依 cfg 建立 dataloader / model / 損失函式 / 優化器 / scheduler"""

    # ============ Datasets / Loaders [NEW] ============
    base = datasets.ImageFolder(root=data_root / "all_data") # 不綁 transform
    class_names = base.classes # 類別名稱
    num_classes = len(class_names)
    targets = np.array(base.targets)  # 每張圖的類別 id

    # 目標比例
    train_ratio, val_ratio2, test_ratio = 0.70, 0.15, 0.15

    # 固定切分與 loader 洗牌的種子
    g_split = torch.Generator().manual_seed(seed) # 固定切分
    g_loader = torch.Generator().manual_seed(seed + trial_id*7) # 固定 DataLoader 洗牌
    
    # 切分檔名
    splits_path = (Path("splits") / "es6_70_15_15.json")

    if splits_path.exists(): # 若有舊檔就沿用
        d = json.load(open(splits_path, "r"))
        train_idx, val_idx, test_idx = d["train_idx"], d["val_idx"], d["test_idx"]
        assert len(train_idx) + len(val_idx) + len(test_idx) == len(base)
    else: # 否則就切分並存檔
        train_idx, val_idx, test_idx = [], [], [] # 之後會存起來
        for c in range(num_classes):
            cls_idx = np.where(targets == c)[0]
            # 打散（可重現）
            perm = torch.randperm(len(cls_idx), generator=g_split).numpy()
            cls_idx = cls_idx[perm]

            n = len(cls_idx)
            n_train = int(round(n * train_ratio))
            n_val   = int(round(n * val_ratio2))
            # 剩下的給 test，避免加總四捨五入誤差
            n_test  = n - n_train - n_val 

            train_idx.extend(cls_idx[:n_train].tolist())
            val_idx.extend(cls_idx[n_train:n_train+n_val].tolist())
            test_idx.extend(cls_idx[n_train+n_val:].tolist())

        
        # 存檔以利重現
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump({"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}, f, ensure_ascii=False, indent=2)

    # 切分完成後列一次每類別的數量表，方便寫報告附錄：
    from collections import Counter
    def split_count(idx):
        return Counter([targets[i] for i in idx])

    cnt_tr, cnt_va, cnt_te = split_count(train_idx), split_count(val_idx), split_count(test_idx)
    print("\n[COUNT] per-class sizes:")
    for c, name in enumerate(class_names):
        print(f"{name:>8s}  train={cnt_tr.get(c,0):4d}  val={cnt_va.get(c,0):4d}  test={cnt_te.get(c,0):4d}")


    # 3) 為 train/val/test 建立帶不同 transform 的 Dataset + Subset
    transform_train, transform_eval = make_transforms( # 讀 cfg 覆蓋
        img_size=cfg.get("img_size", img_size),
        rotation_deg=cfg.get("rotation_deg", 10),
    )
    train_base = datasets.ImageFolder(root=base.root , transform=transform_train) 
    eval_base  = datasets.ImageFolder(root=base.root , transform=transform_eval) # val/test 共用
    assert (data_root / "all_data").exists(), "請先把 train+test 合併到 all_data/"


    from torch.utils.data import Subset
    train_ds = Subset(train_base, train_idx)
    val_ds   = Subset(eval_base,  val_idx) 
    test_ds  = Subset(eval_base,  test_idx)

    # 4) Dataloaders（train 洗牌、val/test 不洗牌）
    num_workers = min(8, max(2, (os.cpu_count() or 4)//2))
    pin_memory  = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=g_loader, num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=seed_worker,) # [NEW] 增加 worker_init_fn augmentation、多worker載資料、多次重跑，重現性會比現在好 0309
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=seed_worker) # [NEW] 0309
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=seed_worker) # [NEW] 0309

    assert train_base.class_to_idx == eval_base.class_to_idx == base.class_to_idx

    print(f"[INFO] split sizes = train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)}")
    

    # ============ Model（ResNet34） ============
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features # 最後一層的輸入特徵數
    model.fc = nn.Linear(in_features, num_classes) # 換成符合本任務的輸出

# [NEW] 0927：類別權重與 Label Smoothing
    # === Transfer mode 開關（"finetune" 或 "feature"）===
    # transfer_mode = cfg.get("transfer_mode", "finetune")
    # 一律 fine-tune（不支援 feature 模式）

    # Fine-Tuning：預設全部解凍；Feature 模式則只開 fc（凍結 backbone）
    for name, p in model.named_parameters():
        p.requires_grad = True
    # if transfer_mode == "feature":
        # for name, p in model.named_parameters():
        #     if not name.startswith("fc."):
        #         p.requires_grad = False  # 凍結 backbone

    model = model.to(device)

    # ============ Loss / Optimizer / Scheduler ============
    oval_w = cfg.get("oval_weight", 1.15)  # 允許窮舉調整（原本是 1.15）
    weights_vec = torch.ones(num_classes, dtype=torch.float)
    lower_names = [c.lower() for c in class_names]
    if "oval" in lower_names:
        weights_vec[lower_names.index("oval")] = oval_w
    else:
        print("[WARN] 找不到類別 'Oval'，所有類別權重維持 1.0")
    weights_vec = weights_vec.to(device)
    ls = cfg.get("label_smoothing", 0.1)   # 允許窮舉調整
    criterion = nn.CrossEntropyLoss(label_smoothing=ls, weight=weights_vec) # 用類別權重與Label Smoothing的交叉熵使用 class weight 降低過擬合風險
    
    # ============ Optimizer / Scheduler ============
    this_lr = cfg.get("lr", lr)   
    this_wd = cfg.get("weight_decay", weight_decay)  

    # 「只收可訓練參數（本版一律 fine-tune）」以避免歷史殘影
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_update, lr=this_lr, weight_decay=this_wd) # [NEW] AdamW 0927 用 AdamW 訓練（這裡 params_to_update 等同全模型參數，因為上面全都 requires_grad=True）。    

    # # 以 **val_acc** 當監控指標（若改成 val_macroF1，需先計算後再 scheduler.step(val_macroF1)）[NEW] 1012
    plateau_patience = cfg.get("plateau_patience", 3)  # NEW
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=plateau_patience, factor=0.5, cooldown=1
    )
        
    return {
        "transform_eval": transform_eval,  # 測試/繪圖可能會用到 這是驗證/測試用的 transforms
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "class_names": class_names,
        "num_classes": num_classes,
        "img_size": cfg.get("img_size", img_size),
        # "transfer_mode": transfer_mode
    }
# ============ 訓練迴圈（含早停） ============
def run_epoch(dataloader, phase, model, criterion, optimizer):  # 顯式傳入物件
    if phase == "train":
        model.train() 
    else:
        model.eval()  
    
    running_loss = 0.0
    running_corrects = 0
    n = 0

    for x, y in dataloader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if phase == "train":
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(phase == "train"):
            out = model(x)
            loss = criterion(out, y)
            preds = out.argmax(1)

            if phase == "train":
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * x.size(0)
        running_corrects += (preds == y).sum().item()
        n += x.size(0)
        
    epoch_loss = running_loss / n if n > 0 else 0.0
    epoch_acc  = running_corrects / n if n > 0 else 0.0
    return epoch_loss, epoch_acc

def evaluate_loader_metrics(model, dataloader, class_names):
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(x)
            preds = out.argmax(1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    from sklearn.metrics import accuracy_score, classification_report

    acc = accuracy_score(all_labels, all_preds)
    rep = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4,
        output_dict=True
    )
    macro_f1 = float(rep["macro avg"]["f1-score"])

    return acc, macro_f1

# ============ 跑「單一組合」的完整流程（訓練→挑 best→測試） ============
def train_one_combo(cfg, trial_id):  # 針對單一組合跑完整流程

    reseed_all(seed + trial_id * 1000) # 讓同一組 cfg 在任意 trial 順序都用到同一條隨機序列
    pack = build_data_and_model(cfg, trial_id) # 依 cfg 建立所有東西

    # 1. 資料載入器
    train_loader = pack["train_loader"]
    val_loader   = pack["val_loader"]
    test_loader  = pack["test_loader"]

    # 2. 建模相關
    model        = pack["model"]
    criterion    = pack["criterion"]
    optimizer    = pack["optimizer"]
    scheduler    = pack["scheduler"]
    class_names  = pack["class_names"]
    num_classes  = pack["num_classes"]
    # transfer_mode= pack.get("transfer_mode", "finetune")

    # 3. 圖表與報表相關
    class_names = pack["class_names"] # 報表與圖表（classification report、混淆矩陣、per-class F1/ROC 的標籤），也用來找 oval 再套 class weight
    num_classes = pack["num_classes"] # 決定 model.fc 的輸出維度、CrossEntropyLoss(weight=...) 的向量長度、以及很多地方的 shape 檢查。
    cur_img_size = pack["img_size"]   # 記錄本次實驗的輸入尺寸

    # 4. 檢查列印 [NEW] 0927
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    print(f"[CHECK] trainable_params={trainable} | num_classes={num_classes}")


    best_w = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    bad = 0

    # 預設 12 輪耐心 [NEW] 1012
    patience_es = cfg.get("earlystop_patience", patience)

    # 每個組合單獨存一個 ckpt（暫存）
   # [FIX] 建立子資料夾再存
    ckpt_dir = Path("checkpoints") / "resnet34_faceshape_best_ES_6"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"resnet34_faceshape_best_ES_6_{trial_id}.pth"

    print(f"Start training for {epochs} epochs on {device} ...")
    history = {"epoch": [], "train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []} # 記錄訓練過程

    t0 = time.time() 

    try:
        for epoch in range(1, epochs+1):
            reseed_all(seed + trial_id * 1000 + epoch)
            train_loss, train_acc = run_epoch(train_loader, "train", model, criterion, optimizer) 
            val_loss,   val_acc   = run_epoch(val_loader,   "val",   model, criterion, optimizer) 

            scheduler.step(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_w = copy.deepcopy(model.state_dict())
                bad = 0
                torch.save({
                    "state_dict": best_w,
                    "class_names": class_names,
                    "img_size": cur_img_size, # [FIX] 改這行
                }, ckpt_path)
            else:
                bad += 1

            print(f"Epoch {epoch:02d}/{epochs} | "
                f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
                f"best_val_acc={best_acc:.4f} | bad={bad}/{patience_es}")
            
            # 紀錄訓練過程
            history["epoch"].append(epoch) 
            history["train_acc"].append(train_acc) 
            history["val_acc"].append(val_acc) 
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)


            if bad >= patience_es:
                print("Early stopping triggered.")
                break
    except KeyboardInterrupt:
        print("\n[INFO] 手動中止，保存目前最佳權重。")
        torch.save({"state_dict": best_w, 
                    "class_names": class_names, 
                    "img_size": cur_img_size}, # [FIX] 改這行
                    ckpt_path)

    print(f"Training done in {(time.time()-t0):.1f}s. Best val acc={best_acc:.4f}")


# ============ 載入最佳權重後做最終評估 ============
    model.load_state_dict(best_w)
    model.eval()
    # 1) validation 指標（用來選最佳組合）
    val_acc_final, val_macro_f1 = evaluate_loader_metrics(model, val_loader, class_names)

    # 2) test 指標（只做最終報告）
    test_acc, test_macro_f1 = evaluate_loader_metrics(model, test_loader, class_names)

    print(f"Final Val  accuracy : {val_acc_final:.4f}")
    print(f"Final Val  macro_f1 : {val_macro_f1:.4f}")
    print(f"Final Test accuracy : {test_acc:.4f}")
    print(f"Final Test macro_f1 : {test_macro_f1:.4f}")
    return {
        "val_acc": float(val_acc_final),
        "val_macro_f1": float(val_macro_f1),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "best_w": best_w,
        "history": history,
        "class_names": class_names,
        "img_size": cur_img_size,
        "test_loader": test_loader,
    }

def save_trial_artifacts(rec, trial_id, cfg):  # 針對單一 trial 輸出完整成果
    """
    [作用] 針對每個 trial（超參數組合）輸出：
      - acc/loss 曲線（來自 rec['history']）
      - 混淆矩陣（原始 / 正規化）
      - ROC 曲線（OvR）
      - classification_report.txt
      - config.json（紀錄本 trial 超參數）
    [為什麼要做] 讓每組都能展示，不用只看最佳組合。
    """
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    from sklearn.preprocessing import label_binarize
    from torchvision import models

    class_names = rec["class_names"]
    num_classes = len(class_names)

    # === 建立輸出資料夾（含 trial id 與關鍵超參數，方便回溯） ===
    tag = f"trial_{trial_id}_lr{cfg['lr']}_wd{cfg['weight_decay']}_oval{cfg['oval_weight']}_rot{cfg['rotation_deg']}"
    outdir = Path("reports") / "resnet34_ES_6" / tag  # 一組一個資料夾
    outdir.mkdir(parents=True, exist_ok=True)

    # 存一份 config 方便復現
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # === 重建模型、載入該 trial 的最佳權重，做測試推論 ===
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)  # 與訓練一致的 backbone
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)
    model.load_state_dict(rec["best_w"])
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in rec["test_loader"]:
            x = x.to(device, non_blocking=True)
            out = model(x)
            probs = F.softmax(out, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(out.argmax(1).cpu().tolist())
            all_labels.extend(y.tolist())
    all_probs = np.concatenate(all_probs, axis=0)

    # === 1) acc/loss 曲線（用該 trial 的 history） ===
    hist = rec["history"]
    if len(hist["epoch"]) > 0:
        # Acc
        plt.figure()
        plt.plot(hist["epoch"], hist["train_acc"], label="Train Acc")
        plt.plot(hist["epoch"], hist["val_acc"],   label="Val Acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy over Epochs")
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir / "acc_curve.png", dpi=150); plt.close()

        # Loss
        plt.figure()
        plt.plot(hist["epoch"], hist["train_loss"], label="Train Loss")
        plt.plot(hist["epoch"], hist["val_loss"],   label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss over Epochs")
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir / "loss_curve.png", dpi=150); plt.close()

    # === 2) 分類報告（文字檔） ===
    rep_txt = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    with open(outdir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(rep_txt)

    # === 3) 混淆矩陣（原始與正規化） ===
    def _plot_cm(cm, title, outpath, normalize=False):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1e-12)
        plt.figure(figsize=(6,5))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(title); plt.colorbar()
        tick = np.arange(num_classes)
        plt.xticks(tick, class_names, rotation=45); plt.yticks(tick, class_names)
        thresh = cm.max() / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                txt = f"{cm[i,j]:.2f}" if normalize else f"{int(cm[i,j])}"
                plt.text(j, i, txt, ha="center",
                         color="white" if cm[i,j] > thresh else "black", fontsize=9)
        plt.ylabel("True label"); plt.xlabel("Predicted label")
        plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    print("Confusion matrix (THIS TRIAL):")
    print(cm)
    # === 新增：把混淆矩陣數字也存起來（原始 + 正規化 + JSON） ===
    np.savetxt(outdir / "confusion_matrix_raw.txt", cm, fmt="%d")
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-12)
    np.savetxt(outdir / "confusion_matrix_norm_raw.txt", cm_norm, fmt="%.4f")
    with open(outdir / "confusion_matrix_raw.json", "w", encoding="utf-8") as f:
        json.dump(cm.tolist(), f, ensure_ascii=False, indent=2)

    # 圖檔（原始 / 正規化）
    _plot_cm(cm, "Confusion Matrix", outdir / "confusion_matrix.png", normalize=False)   # 原始
    _plot_cm(cm, "Confusion Matrix (Normalized)", outdir / "confusion_matrix_norm.png", normalize=True)  # 正規化

    # === 4) ROC（OvR） ===
    y_true = np.array(all_labels)
    y_score = np.array(all_probs)
    y_onehot = label_binarize(y_true, classes=list(range(num_classes)))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # micro/macro
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr; tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(7,6))
    plt.plot(fpr["micro"], tpr["micro"], label=f"micro-avg (AUC={roc_auc['micro']:.3f})", linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"], label=f"macro-avg (AUC={roc_auc['macro']:.3f})", linewidth=2)
    for i, name in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], alpha=0.85, label=f"{name} (AUC={roc_auc[i]:.3f})")
    plt.plot([0,1],[0,1], linestyle="--", linewidth=1)
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC (OvR)"); plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout(); plt.savefig(outdir / "roc_curve.png", dpi=150); plt.close()

    print(f"[SHOWCASE] Artifacts saved to: {outdir}") 

# ========================= 主程式入口 =========================
if __name__ == "__main__":  # 外層從這裡開始
    from itertools import product  # 用來做窮舉組合
    os.makedirs("reports", exist_ok=True) 
    print(f"[CHECK] device={device}")

    # ============ 「小型網格」：先跑這些就能看到方向 ============
    grid = {
        "img_size": [256],
        "lr": [1.4e-4, 1.5e-4, 1.6e-4],
        "weight_decay": [1e-4, 1.2e-4],
        "label_smoothing": [0.1],
        "oval_weight": [1.15, 1.16, 1.17],
        "rotation_deg": [10],
        "earlystop_patience": [20],
        "plateau_patience": [3],
    }

    # ============ 外層窮舉迴圈：逐組跑 ============
    results = []     # 排行榜原始資料
    best_rec = None  
    trial_id = 0     

    for combo in product(*grid.values()):  # 窮舉所有組合
        cfg = dict(zip(grid.keys(), combo))
        trial_id += 1
        print(f"\n[GRID] Trial {trial_id}: {cfg}")
        rec = train_one_combo(cfg, trial_id)  # 訓練＋測試一輪
        rec["cfg"] = cfg
        
        # 立刻輸出這一組 trial 的所有圖表/報告/混淆矩陣（含數字檔）
        save_trial_artifacts(rec, trial_id, cfg)

        # 更新最佳（以 val_acc 為主）
        if (best_rec is None) or (rec["val_acc"] > best_rec["val_acc"]):  
            best_rec = rec

        results.append({
            "val_acc": rec["val_acc"],
            "test_acc": rec["test_acc"],
            "test_macro_f1": rec["test_macro_f1"],
            "cfg": cfg,
        })

        # 釋放不再需要的重量級內容
        del rec
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ============ 排行榜輸出（印出 + 存 CSV） ============
    # 依 val_acc 由高到低排序
    results_sorted = sorted(results, key=lambda r: r.get("val_acc", 0.0), reverse=True)
    print("\n========== BEST RESULT ==========")
    print(f"test_macro_f1 : {best_rec['test_macro_f1']:.4f}")
    print(f"val_acc       : {best_rec['val_acc']:.4f}")
    print(f"test_acc      : {best_rec['test_acc']:.4f}")
    print(f"config        : {best_rec['cfg']}")

    # 印簡表
    print("\n[GRID][LEADERBOARD] (Top 10)")
    for i, r in enumerate(results_sorted[:10], 1):
        print(f"{i:2d}. test_macro_f1={r.get('test_macro_f1', float('nan')):.4f} | val_acc={r['val_acc']:.4f} | test_acc={r['test_acc']:.4f} | cfg={r['cfg']}")

    # 存成 CSV
    import csv
    csv_path = Path("reports") / "grid_results_resnet34_ES_6.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "val_acc", "test_acc", "test_macro_f1",
            "img_size", "lr", "weight_decay", "label_smoothing", "oval_weight", "rotation_deg"
        ])
        for r in results_sorted:
            c = r["cfg"]
            writer.writerow([
                r["val_acc"],
                r["test_acc"],
                f"{r.get('test_macro_f1', float('nan')):.6f}",
                c["img_size"],
                c["lr"],
                c["weight_decay"],
                c["label_smoothing"],
                c["oval_weight"],
                c["rotation_deg"],
            ])

    best_txt = os.path.join("reports", "best_result_summary.txt")

    with open(best_txt, "w", encoding="utf-8") as f:
        f.write("===== BEST RESULT =====\n")
        f.write(f"test_macro_f1 : {best_rec['test_macro_f1']:.4f}\n")
        f.write(f"val_acc  : {best_rec['val_acc']:.4f}\n")
        f.write(f"test_acc : {best_rec['test_acc']:.4f}\n")
        f.write(f"config   : {best_rec['cfg']}\n")
        f.write(f"csv_path : {csv_path.resolve()}\n")

    print(f"[GRID] 排行榜 CSV 已輸出：{csv_path.resolve()}")
    print(f"[GRID] 最佳結果摘要已輸出：{Path(best_txt).resolve()}")

    # 存最佳設定（可重現）
    best_cfg_path = Path("reports") / "best_config_resnet34_ES_6.json" 
    with open(best_cfg_path, "w", encoding="utf-8") as f:
        json.dump(best_rec["cfg"], f, ensure_ascii=False, indent=2)
    print(f"[GRID] 最佳組合設定已輸出：{best_cfg_path.resolve()}")

    
    # ============ 用「最佳組合」產出完整報表與圖表（你原本的流程） ============
    print("\n[GRID] 以最佳組合產生完整報表與圖表 ...")

    # 重新構建模型並載入最佳權重（確保乾淨）
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)  # [NEW]
    in_features = model.fc.in_features                                      # [NEW]
    model.fc = nn.Linear(in_features, len(best_rec["class_names"]))         # [NEW]
    model = model.to(device)                                                # [NEW]
    model.load_state_dict(best_rec["best_w"])                               # [NEW]
    model.eval()                                                            # [NEW]

    # 重新跑 test，拿到完整報告
    import torch.nn.functional as F
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in best_rec["test_loader"]:
            x = x.to(device, non_blocking=True)
            out = model(x)
            probs = F.softmax(out, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds += out.argmax(1).cpu().tolist()
            all_labels += y.tolist()
    all_probs = np.concatenate(all_probs, axis=0)

    print("\nClassification report (BEST CONFIG):")
    print(classification_report(all_labels, all_preds, target_names=best_rec["class_names"], digits=4))

    # 另存最佳模型（對外發佈用）
    ckpt_best = Path("checkpoints") / "resnet34_faceshape_best_grid_ES_6.pth"   # [NEW]
    torch.save({"state_dict": best_rec["best_w"], 
                "class_names": best_rec["class_names"], 
                "img_size": best_rec["img_size"]}, ckpt_best)
    print(f"\nBest checkpoint saved to: {ckpt_best.resolve()}")


    # ===== 生成圖表到 ./reports =====
    import matplotlib.pyplot as plt

    os.makedirs("reports", exist_ok=True) # 先確保有 reports/ 資料夾
    # 建立專屬資料夾
    report_dir = Path("reports") / "resnet34_ES_6"  # 再指定子資料夾 reports/resnet34
    report_dir.mkdir(parents=True, exist_ok=True) # 建立 reports/resnet34

    # Acc / Loss 曲線
    # 1) 曲線：用最佳 trial 的 history
    history = best_rec["history"]
    if len(history["epoch"]) > 0:
        # Accuracy
        plt.figure()
        plt.plot(history["epoch"], history["train_acc"], label="Train Acc")
        plt.plot(history["epoch"], history["val_acc"],   label="Val Acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy over Epochs")
        plt.tight_layout(); plt.savefig(report_dir / "acc_curve.png", dpi=150) # 儲存準確率曲線

        # Loss
        plt.figure()
        plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
        plt.plot(history["epoch"], history["val_loss"],   label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss over Epochs")
        plt.tight_layout(); plt.savefig(report_dir / "loss_curve.png", dpi=150) # 儲存損失曲線

    # ====== 多分類 ROC 曲線（OvR：One-vs-Rest） ======
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    class_names = best_rec["class_names"] # ['Heart','Oblong','Oval','Round','Square']
    num_classes = len(class_names) # 5

    y_true = np.array(all_labels)                                # [N]
    y_score = np.array(all_probs)                                # [N, C]
    y_onehot = label_binarize(y_true, classes=list(range(num_classes)))  # [N, C]

    # 每一類的 ROC 與 AUC
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro-average（把所有類當一個二分類）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro-average（所有類的 FPR 串聯取唯一點，再取平均 TPR）
    # 收集所有 fpr 後取聯集
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    # 在這些點插值計算每類 tpr，再平均
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 繪圖
    plt.figure(figsize=(7,6))
    # micro / macro
    plt.plot(fpr["micro"], tpr["micro"],
            label=f"micro-avg ROC (AUC = {roc_auc['micro']:.3f})", linewidth=2)
    plt.plot(fpr["macro"], tpr["macro"],
            label=f"macro-avg ROC (AUC = {roc_auc['macro']:.3f})", linewidth=2)

    # 每個類別
    for i, name in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], alpha=0.8, label=f"{name} (AUC = {roc_auc[i]:.3f})")

    plt.plot([0,1],[0,1], linestyle="--", linewidth=1)  # 參考虛線
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC (OvR)")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(report_dir / "roc_curve.png", dpi=150)
    print("已輸出：", report_dir / "roc_curve.png")


    # 取得字典版報告 各類 Precision / Recall / F1 與 Overall Accuracy（最佳組合）
    rep = classification_report(all_labels, all_preds, target_names=class_names, digits=4, output_dict=True)

    macro_f1 = float(rep["macro avg"]["f1-score"]) # 取出 macro f1 [NEW] 0927

    # 各類 precision 精確 / recall 召回 / f1-score
    for m in ["precision", "recall", "f1-score"]:
        vals = [rep[name][m] for name in class_names]
        plt.figure()
        plt.bar(range(len(class_names)), vals)
        plt.xticks(range(len(class_names)), class_names)
        plt.ylim(0, 1.0)
        plt.ylabel(m.capitalize()); plt.title(f"{m.capitalize()} by Class")
        plt.tight_layout(); plt.savefig(report_dir / f"{m}_by_class.png", dpi=150)

    # Overall accuracy 整體準確率
    overall_acc = rep["accuracy"]
    plt.figure()
    plt.bar([0],[overall_acc]); plt.xticks([0],["accuracy"]); plt.ylim(0,1.0)
    plt.title(f"Overall Accuracy: {overall_acc:.3f}")
    plt.tight_layout(); plt.savefig(report_dir / "overall_accuracy.png", dpi=150)

    # 混淆矩陣
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(best_rec["class_names"]))))
    # print("Class order:", class_names)            # 或 best_rec["class_names"]
    # print("CM shape:", cm.shape)
    print("Confusion matrix (BEST CONFIG):")
    print(cm)
    # 把最佳組合的混淆矩陣存成數字
    np.savetxt(report_dir / "confusion_matrix_best_raw_es_6.txt", cm, fmt="%d")
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-12)
    np.savetxt(report_dir / "confusion_matrix_best_norm_raw_es_6.txt", cm_norm, fmt="%.4f")
    with open(report_dir / "confusion_matrix_best_raw_es_6.json", "w", encoding="utf-8") as f:
        json.dump(cm.tolist(), f, ensure_ascii=False, indent=2)

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap="Blues") # cm是矩陣 interpolation是插值 cmap是顏色
    plt.title("Confusion Matrix"); plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45); plt.yticks(ticks, class_names)
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout(); plt.savefig(report_dir / "confusion_matrix.png", dpi=150)

    print("\n[INFO] 最佳組合的圖表已輸出到 ./reports/resnet34_ES_6 ： ")
    print("  acc_curve.png, loss_curve.png, precision_by_class.png, recall_by_class.png, f1-score_by_class.png, overall_accuracy.png, confusion_matrix.png")

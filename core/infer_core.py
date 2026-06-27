# 推論核心（程式邏輯抽出放這，便於單元測試）
# infer_core.py

from __future__ import annotations


import math
from pathlib import Path
import random
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from core import thresholds as T
from utils.recommendation_engine import recommend_hairstyles, format_recommendation_text


def _build_inference_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])



    """用臉框建立 68 點近似 landmark，避免依賴 mediapipe solutions。"""

    def scale_points(points: np.ndarray) -> np.ndarray:
        out = np.empty_like(points, dtype=np.float32)
        out[:, 0] = x + points[:, 0] * w
        out[:, 1] = y + points[:, 1] * h
        return out

    points = np.zeros((68, 2), dtype=np.float32)

    # 臉輪廓 0~16
    jaw_t = np.linspace(math.pi, 2 * math.pi, 17)
    points[0:17, 0] = 0.5 + 0.42 * np.cos(jaw_t)
    points[0:17, 1] = 0.70 + 0.28 * np.sin(jaw_t)

    # 左眉 17~21
    left_brow_x = np.linspace(0.30, 0.46, 5)
    left_brow_y = np.array([0.27, 0.25, 0.24, 0.25, 0.27], dtype=np.float32)
    points[17:22] = np.column_stack([left_brow_x, left_brow_y])

    # 右眉 22~26
    right_brow_x = np.linspace(0.54, 0.70, 5)
    right_brow_y = np.array([0.27, 0.25, 0.24, 0.25, 0.27], dtype=np.float32)
    points[22:27] = np.column_stack([right_brow_x, right_brow_y])

    # 鼻樑與鼻頭 27~35
    points[27:31] = np.column_stack([
        np.full(4, 0.50, dtype=np.float32),
        np.linspace(0.33, 0.56, 4, dtype=np.float32),
    ])
    points[31:36] = np.array([
        [0.42, 0.58],
        [0.46, 0.60],
        [0.50, 0.61],
        [0.54, 0.60],
        [0.58, 0.58],
    ], dtype=np.float32)

    # 眼睛 36~47
    left_eye_t = np.linspace(0, 2 * math.pi, 6, endpoint=False)
    right_eye_t = np.linspace(0, 2 * math.pi, 6, endpoint=False)
    points[36:42, 0] = 0.36 + 0.08 * np.cos(left_eye_t)
    points[36:42, 1] = 0.41 + 0.05 * np.sin(left_eye_t)
    points[42:48, 0] = 0.64 + 0.08 * np.cos(right_eye_t)
    points[42:48, 1] = 0.41 + 0.05 * np.sin(right_eye_t)

    # 嘴巴 48~67
    mouth_outer_t = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    mouth_inner_t = np.linspace(0, 2 * math.pi, 8, endpoint=False)
    points[48:60, 0] = 0.50 + 0.15 * np.cos(mouth_outer_t)
    points[48:60, 1] = 0.73 + 0.08 * np.sin(mouth_outer_t)
    points[60:68, 0] = 0.50 + 0.08 * np.cos(mouth_inner_t)
    points[60:68, 1] = 0.73 + 0.04 * np.sin(mouth_inner_t)

    points[:, 0] = np.clip(points[:, 0], 0.02, 0.98)
    points[:, 1] = np.clip(points[:, 1], 0.02, 0.98)
    return scale_points(points)


def load_model(ckpt_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        class_names = ckpt.get("class_names", ["Heart", "Oblong", "Oval", "Round", "Square"])
        img_size = int(ckpt.get("img_size", 256))
    else:
        state_dict = ckpt
        class_names = ["Heart", "Oblong", "Oval", "Round", "Square"]
        img_size = 256

    model = models.resnet34(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(class_names))
    model.load_state_dict(state_dict)
    model.eval()

    tf = _build_inference_transform(img_size)
    return model, class_names, img_size, tf



# ===================== 讀圖與偵測 =====================

def pil_to_bgr(img: Image.Image):
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def detect_landmarks(img_bgr, use_haar: bool = False, haar_path: Path | None = None):
    """
    回傳 (rect, landmarks)
    - 預設使用 MediaPipe FaceMesh 抓真實臉部關鍵點
    - use_haar=True 時只用 Haar 偵測臉框，並回傳 None landmarks
    """

    h, w = img_bgr.shape[:2]

    if not use_haar:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh_detector:
            result = face_mesh_detector.process(img_rgb)

        if not result.multi_face_landmarks:
            return None, None

        face_landmarks = result.multi_face_landmarks[0]
        pts = np.array(
            [[lm.x * w, lm.y * h] for lm in face_landmarks.landmark],
            dtype=np.float32
        )

        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        rect = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

        return rect, pts

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = haar_path or Path("models/haarcascade_frontalface_default.xml")

    if not cascade_path.exists():
        return None, None

    cascade = cv2.CascadeClassifier(str(cascade_path))
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None, None

    x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
    return (int(x), int(y), int(fw), int(fh)), None


# ===================== 對齊：眼睛水平 + 方形裁切 =====================

def align_face(img_bgr, landmarks: np.ndarray | None, rect=None, margin: float = 0.25):
    h, w = img_bgr.shape[:2]

    if landmarks is None:
        x, y, fw, fh = rect
        pad = int(max(fw, fh) * margin)

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + fw + pad)
        y2 = min(h, y + fh + pad)

        crop = img_bgr[y1:y2, x1:x2]
        return crop, img_bgr, None

    left_eye = landmarks[[33, 133]].mean(axis=0)
    right_eye = landmarks[[362, 263]].mean(axis=0)

    dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img_rot = cv2.warpAffine(
        img_bgr,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    ones = np.ones((landmarks.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([landmarks, ones])
    M3 = np.vstack([M, [0, 0, 1]])
    pts_rot = (M3 @ pts_h.T).T[:, :2]

    x1, y1 = pts_rot.min(axis=0)
    x2, y2 = pts_rot.max(axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1.0 + margin * 2)

    x1s, y1s = int(round(cx - side / 2)), int(round(cy - side / 2))
    x2s, y2s = int(round(cx + side / 2)), int(round(cy + side / 2))

    padL, padT = max(0, -x1s), max(0, -y1s)
    padR, padB = max(0, x2s - w), max(0, y2s - h)

    if padL or padT or padR or padB:
        img_rot = cv2.copyMakeBorder(
            img_rot,
            padT,
            padB,
            padL,
            padR,
            cv2.BORDER_REPLICATE
        )
        x1s += padL
        x2s += padL
        y1s += padT
        y2s += padT
        pts_rot[:, 0] += padL
        pts_rot[:, 1] += padT

    crop = img_rot[y1s:y2s, x1s:x2s]
    return crop, img_rot, pts_rot


# ===================== 幾何量與補票 =====================

def geometry_measures(landmarks: np.ndarray, face_top_y: int):
    cheekbone_width = np.linalg.norm(landmarks[454] - landmarks[234])
    jaw_width = np.linalg.norm(landmarks[172] - landmarks[397])
    face_length = landmarks[152][1] - face_top_y
    L_forehead = np.linalg.norm(landmarks[103] - landmarks[332])

    a, b, c, d = landmarks[172], landmarks[136], landmarks[150], landmarks[176]
    alpha0 = math.degrees(math.atan2(c[1] - a[1], c[0] - a[0]))
    alpha1 = math.degrees(math.atan2(d[1] - b[1], d[0] - b[0]))
    angle = 180 - abs(alpha1 - alpha0)

    return face_length, cheekbone_width, jaw_width, L_forehead, angle
def geometry_vote(landmarks: np.ndarray, face_top_y: int, classes: list[str]):
    L_len, L_cheek, L_jaw, L_forehead, jaw_angle = geometry_measures(landmarks, face_top_y)
    r_len_cheek  = L_len / (L_cheek + 1e-6)
    r_jaw_cheek  = L_jaw / (L_cheek + 1e-6)
    r_fore_cheek = L_forehead / (L_cheek + 1e-6)

    msg = (f"L2={L_len:.1f}, L3={L_cheek:.1f}, L4={L_jaw:.1f}, "
           f"L5={L_forehead:.1f}, 下顎角≈{jaw_angle:.0f}° | "
           f"len/cheek={r_len_cheek:.2f}, jaw/cheek={r_jaw_cheek:.2f}, fore/cheek={r_fore_cheek:.2f}")

    # 1) 長臉
    if r_len_cheek > T.OBLONG_R_LEN_CHEEK:
        return ("Oblong" if "Oblong" in classes else classes[0],
                f"{msg}；長/顴>{T.OBLONG_R_LEN_CHEEK:.2f} →Oblong")

    # 2) 方臉
    if (T.SQUARE_R_JAW_CHEEK_MIN <= r_jaw_cheek <= T.SQUARE_R_JAW_CHEEK_MAX) and \
       (jaw_angle < T.SQUARE_JAW_ANGLE_MAX):
        return ("Square" if "Square" in classes else classes[0],
                f"{msg}；下顎/顴≈1 且角度<{T.SQUARE_JAW_ANGLE_MAX}° →Square")

    # 3) 心形（上寬下窄）
    if r_jaw_cheek < T.SQUARE_R_JAW_CHEEK_MIN:
        return ("Heart" if "Heart" in classes else classes[0],
                f"{msg}；下顎/顴<{T.SQUARE_R_JAW_CHEEK_MIN:.2f} →Heart")

    # 4) 鈍角：Round / Oval
    if jaw_angle >= T.ROUND_JAW_ANGLE_MIN:
        if abs(L_len - L_cheek) < T.ROUND_LEN_CHEEK_DELTA * L_cheek:
            return ("Round" if "Round" in classes else classes[0],
                    f"{msg}；角度≥{T.ROUND_JAW_ANGLE_MIN}° 且長寬接近→Round")
        else:
            return ("Oval" if "Oval" in classes else classes[0],
                    f"{msg}；角度≥{T.ROUND_JAW_ANGLE_MIN}° 且略長→Oval")

    # 5) 其餘：略長且比例平衡 → Oval
    return ("Oval" if "Oval" in classes else classes[0], f"{msg}；比例平衡→Oval")


# ===================== 推論一次（核心） =====================

def infer_once(image_pil: Image.Image,
               model: torch.nn.Module,
               class_names: list[str],
               tf,
               predictor: Any | None = None,
               use_haar: bool = False,
               haar_path: Path | None = None,
               low_conf: float = 0.55):
    """回傳 dict：{vis_pil, top1, p1, top2, p2, tips, gallery_cls, used_geo, geo_msg}，其中 gallery_cls 用於選擇圖片資料夾。"""
    img_bgr = pil_to_bgr(image_pil)
    rect, pts = detect_landmarks(img_bgr, use_haar=use_haar, haar_path=haar_path)
    if rect is None:
        return {"error": "偵測不到人臉，請換正臉、光線均勻的照片"}

    crop, img_rot, pts_rot = align_face(img_bgr, pts, rect=rect, margin=0.25)
    face_top_y = max(0, int(pts_rot[:, 1].min())) if pts_rot is not None else None
    
    geometry_info = None # 初始化 geometry_info 為 None

    if pts_rot is not None:
        L_len, L_cheek, L_jaw, L_fore, jaw_angle = geometry_measures(pts_rot, face_top_y)

        geometry_info = {
            "face_length_ratio": float(L_len / (L_cheek + 1e-6)),
            "jaw_width_ratio": float(L_jaw / (L_cheek + 1e-6)),
            "forehead_width_ratio": float(L_fore / (L_cheek + 1e-6)),
            "jaw_angle": float(jaw_angle),
        }

    pil_face = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    x = tf(pil_face).unsqueeze(0)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0].cpu().numpy()

    order = prob.argsort()[::-1]
    top1_idx, top2_idx = order[0], order[1]
    top1, p1 = class_names[top1_idx], float(prob[top1_idx])
    top2, p2 = class_names[top2_idx], float(prob[top2_idx])

    prob_dict = {
        class_names[i]: float(prob[i])
        for i in range(len(class_names))
    }

    # # === 強覆蓋 Square（直接下判斷）===
    # L_len, L_cheek, L_jaw, L_fore, jaw_angle = geometry_measures(pts_rot, face_top_y)
    # r_jaw_cheek = L_jaw / (L_cheek + 1e-6)
    # if (jaw_angle < T.SQUARE_JAW_ANGLE_MAX) and (r_jaw_cheek >= T.SQUARE_R_JAW_CHEEK_MIN):
    #     top2, p2 = top1, p1
    #     top1, p1 = "Square", 0.99
    # # === 強覆蓋結束 ===


    used_geo, geo_msg = False, ""
    if p1 < low_conf and pts_rot is not None:
        geo_cls, geo_msg = geometry_vote(pts_rot, face_top_y, class_names)
        used_geo = True
        fuse = prob.copy()
        j = class_names.index(geo_cls)
        onehot = np.zeros_like(fuse); onehot[j] = 1.0
        fuse = 0.7 * fuse + 0.3 * onehot
        order = fuse.argsort()[::-1]
        top1, p1 = class_names[order[0]], float(fuse[order[0]])
        top2, p2 = class_names[order[1]], float(fuse[order[1]])

        prob_dict = {
            class_names[i]: float(fuse[i])
            for i in range(len(class_names))
        }

    vis = crop.copy()

    if pts_rot is not None:
        x1, y1 = pts_rot.min(axis=0)
        x2, y2 = pts_rot.max(axis=0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1) * 1.5
        x1s, y1s = int(round(cx - side / 2)), int(round(cy - side / 2))

        pts_crop = pts_rot.copy()
        pts_crop[:, 0] -= x1s
        pts_crop[:, 1] -= y1s

        for x, y in pts_crop.astype(int):
            if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)

    vis_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    rec_result = recommend_hairstyles(top1, top_k=6)
    tips = format_recommendation_text(rec_result)

    return {
        "vis_pil": vis_pil,
        "top1": top1, "p1": p1,
        "top2": top2, "p2": p2,
        "tips": tips,
        "gallery_cls": top1,
        "used_geo": used_geo,
        "geo_msg": geo_msg,
        "prob_dict": prob_dict,
        "geometry_info": geometry_info,
    }


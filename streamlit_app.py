import io, math, tempfile, traceback, os
from pathlib import Path
from typing import Dict, Tuple
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ExifTags

# ====== scikit & joblib ======
from joblib import load as joblib_load

# ====== Mediapipe ======
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles

# =============================
# KONFIGURASI
# =============================
IMG_SIZE = (224, 224)
FEATURE_NAMES = [
    "eye_slant_deg",
    "palpebral_height_over_width",
    "intercanthal_over_outercanthal",
    "nose_width_over_ipd",
    "philtrum_over_nose_height",
    "mouth_open_over_width",
    "face_aspect_h_over_w",
]

# =============================
# INDEKS LANDMARK
# =============================
IDX = {
    "RIGHT_IRIS": [469, 470, 471, 472, 473],
    "LEFT_IRIS" : [474, 475, 476, 477],
    "RIGHT_EYE_OUTER": 33,
    "RIGHT_EYE_INNER": 133,
    "LEFT_EYE_INNER" : 362,
    "LEFT_EYE_OUTER" : 263,
    "RIGHT_EYE_UP"   : 159,
    "RIGHT_EYE_DOWN" : 145,
    "LEFT_EYE_UP"    : 386,
    "LEFT_EYE_DOWN"  : 374,
    "MOUTH_RIGHT"    : 61,
    "MOUTH_LEFT"     : 291,
    "LIP_UP_INNER"   : 13,
    "LIP_DOWN_INNER" : 14,
    "NOSE_TIP"       : 1,
    "SUBNASALE"      : 2,
    "NOSE_ALAR_RIGHT": 97,
    "NOSE_ALAR_LEFT" : 326,
    "NASION"         : 168,
}

# =============================
# UTILITAS IMAGE
# =============================
def fix_exif_orientation_cv2(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    try:
        for orient in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orient]=='Orientation':
                break
        exif = pil._getexif()
        if exif is not None:
            o = exif.get(orient, None)
            if o == 3: pil = pil.rotate(180, expand=True)
            elif o == 6: pil = pil.rotate(270, expand=True)
            elif o == 8: pil = pil.rotate(90, expand=True)
    except Exception:
        pass
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def smart_resize(img, min_side=400, max_side=1400):
    h, w = img.shape[:2]
    s = max(h, w)
    if s > max_side:
        scale = max_side / s
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    elif s < min_side and s > 0:
        scale = min_side / s
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    return img

def rotate_image_and_points(image, points, angle_deg, center):
    M = cv2.getRotationMatrix2D(tuple(center), angle_deg, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    ones = np.ones((points.shape[0], 1))
    pts_h = np.hstack([points, ones])
    rotated_pts = (M @ pts_h.T).T
    return rotated, rotated_pts

def crop_square_by_points(image, points, margin=0.35):
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    w, h = x_max - x_min, y_max - y_min
    cx, cy = (x_min + x_max)/2, (y_min + y_max)/2
    side = max(w, h) * (1.0 + margin)
    x1, y1 = int(cx - side/2), int(cy - side/2)
    x2, y2 = int(cx + side/2), int(cy + side/2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1]-1, x2), min(image.shape[0]-1, y2)
    crop = image[y1:y2, x1:x2]
    pts_crop = points.copy()
    pts_crop[:,0] -= x1
    pts_crop[:,1] -= y1
    return crop, pts_crop

def euclid(a, b): 
    return float(np.linalg.norm(a - b))

def iris_center(landmarks_2d, ids):
    pts = landmarks_2d[ids]
    return pts.mean(axis=0)

# =============================
# FITUR EKSTRAKSI
# =============================
def compute_features_from_facemesh(landmarks_2d):
    right_iris_c = iris_center(landmarks_2d, IDX["RIGHT_IRIS"])
    left_iris_c  = iris_center(landmarks_2d, IDX["LEFT_IRIS"])

    dx, dy = (left_iris_c - right_iris_c)
    eye_slant_deg = math.degrees(math.atan2(dy, dx + 1e-8))
    ipd = euclid(right_iris_c, left_iris_c) + 1e-8

    right_eye_w = euclid(landmarks_2d[IDX["RIGHT_EYE_OUTER"]], landmarks_2d[IDX["RIGHT_EYE_INNER"]]) + 1e-8
    right_eye_h = euclid(landmarks_2d[IDX["RIGHT_EYE_UP"]], landmarks_2d[IDX["RIGHT_EYE_DOWN"]])
    left_eye_w  = euclid(landmarks_2d[IDX["LEFT_EYE_OUTER"]], landmarks_2d[IDX["LEFT_EYE_INNER"]]) + 1e-8
    left_eye_h  = euclid(landmarks_2d[IDX["LEFT_EYE_UP"]], landmarks_2d[IDX["LEFT_EYE_DOWN"]])
    palpebral_mean = 0.5 * (right_eye_h/right_eye_w + left_eye_h/left_eye_w)

    inner_canthal = euclid(landmarks_2d[IDX["RIGHT_EYE_INNER"]], landmarks_2d[IDX["LEFT_EYE_INNER"]])
    outer_canthal = euclid(landmarks_2d[IDX["RIGHT_EYE_OUTER"]], landmarks_2d[IDX["LEFT_EYE_OUTER"]]) + 1e-8
    inter_over_outer = inner_canthal / outer_canthal

    nose_w_over_ipd = euclid(landmarks_2d[IDX["NOSE_ALAR_RIGHT"]], landmarks_2d[IDX["NOSE_ALAR_LEFT"]]) / ipd

    nose_h = euclid(landmarks_2d[IDX["NASION"]], landmarks_2d[IDX["SUBNASALE"]]) + 1e-8
    philtrum_over_nose = euclid(landmarks_2d[IDX["SUBNASALE"]], landmarks_2d[IDX["LIP_UP_INNER"]]) / nose_h

    mouth_w = euclid(landmarks_2d[IDX["MOUTH_RIGHT"]], landmarks_2d[IDX["MOUTH_LEFT"]]) + 1e-8
    mouth_open_over_width = euclid(landmarks_2d[IDX["LIP_UP_INNER"]], landmarks_2d[IDX["LIP_DOWN_INNER"]]) / mouth_w

    x_min, y_min = landmarks_2d.min(axis=0)
    x_max, y_max = landmarks_2d.max(axis=0)
    face_aspect = (y_max - y_min + 1e-8) / (x_max - x_min + 1e-8)

    return {
        "eye_slant_deg": float(eye_slant_deg),
        "palpebral_height_over_width": float(palpebral_mean),
        "intercanthal_over_outercanthal": float(inter_over_outer),
        "nose_width_over_ipd": float(nose_w_over_ipd),
        "philtrum_over_nose_height": float(philtrum_over_nose),
        "mouth_open_over_width": float(mouth_open_over_width),
        "face_aspect_h_over_w": float(face_aspect),
        "ipd_px": float(ipd),
    }

# =============================
# FACE MESH GLOBAL
# =============================
_FACE_MESH = None
_FACE_MESH_ERR = None
try:
    _FACE_MESH = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.6,
    )
except Exception as e:
    _FACE_MESH_ERR = e

# =============================
# DRAW OVERLAY HUD
# =============================
def draw_overlay_hud(img_bgr: np.ndarray, pts: np.ndarray, feats: Dict[str, float]) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()

    # Skala dinamis agar proporsional di semua ukuran
    s  = max(1, int(min(h, w) / 400))
    th = 2 * s           # thickness garis
    r  = 2 * s           # radius titik
    fs = 0.45 * s        # font scale
    fw = max(1, s)       # font thickness
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Warna (B, G, R)
    C_BLUE   = (255, 100,  20)   # baseline mata
    C_GREEN  = ( 80, 255,  80)   # slant segmen
    C_YELLOW = ( 40, 220, 255)   # palpebra (tinggi)
    C_MAG    = (255,   0, 255)   # batang hidung
    C_RED    = ( 10,  10, 255)   # mulut
    C_TEXT   = ( 80, 255,  80)   # teks HUD

    # Titik penting (pakai IDX yang sudah kamu pakai sebelumnya)
    p_oL = tuple(np.round(pts[IDX["LEFT_EYE_OUTER"]]).astype(int))
    p_iL = tuple(np.round(pts[IDX["LEFT_EYE_INNER"]]).astype(int))
    p_iR = tuple(np.round(pts[IDX["RIGHT_EYE_INNER"]]).astype(int))
    p_oR = tuple(np.round(pts[IDX["RIGHT_EYE_OUTER"]]).astype(int))

    p_topR = tuple(np.round(pts[IDX["RIGHT_EYE_UP"]]).astype(int))
    p_botR = tuple(np.round(pts[IDX["RIGHT_EYE_DOWN"]]).astype(int))

    p_nas = tuple(np.round(pts[IDX["NASION"]]).astype(int))
    p_sn  = tuple(np.round(pts[IDX["SUBNASALE"]]).astype(int))

    p_mL  = tuple(np.round(pts[IDX["MOUTH_LEFT"]]).astype(int))
    p_mR  = tuple(np.round(pts[IDX["MOUTH_RIGHT"]]).astype(int))

    # 1) Baseline mata (outer kiri ↔ outer kanan) — biru
    cv2.line(out, p_oL, p_oR, C_BLUE, th)

    # 2) Segmen pendek “slant” di dekat sudut luar mata kanan — hijau
    #    ambil arah baseline, gambar potongan ~50*s px menuju outer kanan
    v = np.array(p_oR) - np.array(p_oL)
    if np.linalg.norm(v) > 1e-6:
        v = v / np.linalg.norm(v)
        seg = int(50 * s)
        a = tuple(np.round(np.array(p_oR) - v * seg).astype(int))
        cv2.line(out, a, p_oR, C_GREEN, th)

    # 3) Tinggi palpebra kanan — kuning
    cv2.line(out, p_topR, p_botR, C_YELLOW, th)
    cv2.circle(out, p_topR, r, C_YELLOW, -1)
    cv2.circle(out, p_botR, r, C_YELLOW, -1)

    # 4) Batang hidung (nasion ↕ subnasale) — magenta
    cv2.line(out, p_nas, p_sn, C_MAG, th)
    cv2.circle(out, p_sn, r+1, C_MAG, -1)

    # 5) Garis mulut — merah
    cv2.line(out, p_mL, p_mR, C_RED, th)

    # 6) HUD teks semi-transparan di kiri-atas
    pad = 8 * s
    lines = [
        f"slant: {feats['eye_slant_deg']:+.3f}",
        f"palpebral: {feats['palpebral_height_over_width']:.3f}",
        f"inter/outer: {feats['intercanthal_over_outercanthal']:.3f}",
        f"nose/ipd: {feats['nose_width_over_ipd']:.3f}",
        f"philtrum/noseh: {feats['philtrum_over_nose_height']:.3f}",
        f"open/width: {feats['mouth_open_over_width']:.3f}",
        f"face h/w: {feats['face_aspect_h_over_w']:.3f}",
    ]

    # Hitung ukuran panel
    txt_w = 0; line_h = int(22 * s)
    for t in lines:
        (tw, _), _ = cv2.getTextSize(t, font, fs, fw)
        txt_w = max(txt_w, tw)
    txt_h = line_h * len(lines) + pad
    x0, y0 = pad, pad
    x1, y1 = x0 + txt_w + 2*pad, y0 + txt_h

    # Panel semi-transparan
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (50, 50, 50), -1)
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)

    # Teks
    y = y0 + pad + int(14 * s)
    for t in lines:
        cv2.putText(out, t, (x0 + pad, y), font, fs, C_TEXT, fw, cv2.LINE_AA)
        y += line_h

    return out


# =============================
# PROCESS IMAGE (ALIGN + CROP)
# =============================
def process_image(image_bgr: np.ndarray, draw_overlay=True):
    if _FACE_MESH_ERR is not None:
        raise RuntimeError("Mediapipe FaceMesh gagal dibuat.")

    img0 = fix_exif_orientation_cv2(image_bgr)
    img0 = smart_resize(img0)

    h, w = img0.shape[:2]
    res = _FACE_MESH.process(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None, None, None
    lm = np.array([[p.x*w, p.y*h] for p in res.multi_face_landmarks[0].landmark], dtype=np.float32)

    r_iris_c = iris_center(lm, IDX["RIGHT_IRIS"])
    l_iris_c = iris_center(lm, IDX["LEFT_IRIS"])
    angle = math.degrees(math.atan2((l_iris_c - r_iris_c)[1], (l_iris_c - r_iris_c)[0] + 1e-8))
    center = (r_iris_c + l_iris_c) / 2.0
    aligned_img, aligned_lm = rotate_image_and_points(img0, lm, -angle, center=center)

    crop, lm_crop = crop_square_by_points(aligned_img, aligned_lm, margin=0.35)
    if crop.size == 0: return None, None, None
    crop_resized = cv2.resize(crop, IMG_SIZE, interpolation=cv2.INTER_AREA)
    scale_x = IMG_SIZE[0] / crop.shape[1]
    scale_y = IMG_SIZE[1] / crop.shape[0]
    lm_final = lm_crop.copy()
    lm_final[:,0] *= scale_x
    lm_final[:,1] *= scale_y

    feats = compute_features_from_facemesh(lm_final)
    overlay = draw_overlay_hud(crop_resized, lm_final, feats) if draw_overlay else crop_resized
    return feats, overlay, lm_final

# =============================
# STREAMLIT UI
# =============================
st.set_page_config(page_title="Anthropometry Extractor + Classifier", layout="wide")
st.title("Ekstraksi Fitur Wajah → Klasifikasi (DownSyndrome vs Healthy) Random Forest (0.84)")

scaler = None
clf = None
if Path("artifacts/scaler.joblib").exists():
    scaler = joblib_load("artifacts/scaler.joblib")
if Path("artifacts/rf_model.joblib").exists():
    clf = joblib_load("artifacts/rf_model.joblib")

tab1, tab2 = st.tabs(["🖼️ Single Image", "🗂️ Batch Images"])

# ========== SINGLE ==========
with tab1:
    image_file = st.file_uploader("Upload gambar", type=["jpg","jpeg","png"])
    if image_file:
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        feats, overlay, _ = process_image(img, draw_overlay=True)
        if feats is None:
            st.error("Wajah tidak terdeteksi.")
        else:
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Overlay")
            df_single = pd.DataFrame([feats])[FEATURE_NAMES]
            st.dataframe(df_single)
            if clf is not None:
                X = df_single.to_numpy(dtype=np.float32)
                if scaler is not None: X = scaler.transform(X)
                ypred = clf.predict(X)[0]
                prob = clf.predict_proba(X)[0][1] if hasattr(clf, "predict_proba") else None
                st.success(f"Prediksi: {'Healthy (1)' if ypred==1 else 'DownSyndrome (0)'}")
                if prob is not None:
                    st.write(f"P(Healthy)= {prob:.3f}")
                    fig_p = plt.figure(figsize=(3,3))
                    plt.bar(["DownSyndrome(0)", "Healthy(1)"], [1-prob, prob])
                    plt.ylim(0,1)
                    st.pyplot(fig_p)

# ========== BATCH ==========
with tab2:
    files = st.file_uploader("Upload banyak gambar", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if files:
        rows=[]
        for f in files:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            feats, overlay, _ = process_image(img, draw_overlay=False)
            if feats: rows.append({"filename":f.name, **feats})
        df = pd.DataFrame(rows)
        if clf is not None:
            X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
            if scaler is not None: X = scaler.transform(X)
            preds = clf.predict(X)
            df["pred"] = preds
            if hasattr(clf,"predict_proba"):
                df["p_healthy"]=clf.predict_proba(X)[:,1]
        st.dataframe(df)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False), "batch_features.csv","text/csv")

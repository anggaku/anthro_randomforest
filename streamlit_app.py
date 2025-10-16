# app.py
import io, math, os, tempfile, traceback
from typing import Dict, Tuple
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ====== Optional: scikit & joblib untuk klasifikasi ======
try:
    from joblib import load as joblib_load
except Exception:
    joblib_load = None

# ====== MediaPipe FaceMesh ======
try:
    import mediapipe as mp
except Exception as e:
    st.error("Mediapipe belum terpasang. Tambahkan ke requirements.txt: mediapipe==0.10.14")
    raise e

mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles

# ====== Kompatibilitas UI Streamlit (versi lama/baru) ======
def show_image(img, caption=None):
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        try:
            st.image(img, caption=caption, use_column_width=True)
        except TypeError:
            st.image(img, caption=caption)

def show_df(df, height=None):
    try:
        st.dataframe(df, use_container_width=True, height=height)
    except TypeError:
        try:
            if height is not None:
                st.dataframe(df, height=height)
            else:
                st.dataframe(df)
        except TypeError:
            st.write(df)

# ====== Landmark index yang dipakai (boleh kalibrasi sesuai dataset-mu) ======
LM = {
    "outer_canthus_L": 33,
    "inner_canthus_L": 133,
    "inner_canthus_R": 362,
    "outer_canthus_R": 263,
    "eyelid_top_R": 159,
    "eyelid_bot_R": 145,
    "eyelid_top_L": 386,
    "eyelid_bot_L": 374,
    "nose_alar_L": 98,
    "nose_alar_R": 327,
    "nasion": 168,
    "subnasale": 2,
    "labiale_superius": 13,
    "labiale_inferius": 14,
    "mouth_corner_L": 61,
    "mouth_corner_R": 291,
}

FEATURE_NAMES = [
    "eye_slant_deg",
    "palpebral_height_over_width",
    "intercanthal_over_outercanthal",
    "nose_width_over_ipd",
    "philtrum_over_nose_height",
    "mouth_open_over_width",
    "face_aspect_h_over_w",
]

# ====== Util ======
def _to_np_landmarks(landmarks, img_w, img_h) -> np.ndarray:
    return np.array([(lm.x * img_w, lm.y * img_h) for lm in landmarks.landmark], dtype=np.float32)

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    dx, dy = (b - a)
    return abs(math.degrees(math.atan2(dy, dx)))

def _eye_center(pts: np.ndarray, i_outer: int, i_inner: int) -> np.ndarray:
    return (pts[i_outer] + pts[i_inner]) / 2.0

def extract_features_from_landmarks(pts: np.ndarray, bbox_hw: Tuple[float,float]) -> Dict[str, float]:
    # Eye slant
    eye_slant_deg = _angle_deg(pts[LM["outer_canthus_L"]], pts[LM["outer_canthus_R"]])
    # Palpebral fissure ratio (mean kedua mata)
    h_R = _dist(pts[LM["eyelid_top_R"]], pts[LM["eyelid_bot_R"]])
    w_R = _dist(pts[LM["outer_canthus_L"]], pts[LM["inner_canthus_L"]])
    h_L = _dist(pts[LM["eyelid_top_L"]], pts[LM["eyelid_bot_L"]])
    w_L = _dist(pts[LM["inner_canthus_R"]], pts[LM["outer_canthus_R"]])
    palpebral_height_over_width = float(np.mean([h_R / (w_R + 1e-9), h_L / (w_L + 1e-9)]))
    # Inter/Outer canthal
    intercanthal = _dist(pts[LM["inner_canthus_L"]], pts[LM["inner_canthus_R"]])
    outercanthal = _dist(pts[LM["outer_canthus_L"]], pts[LM["outer_canthus_R"]])
    intercanthal_over_outercanthal = intercanthal / (outercanthal + 1e-9)
    # IPD proxy
    cL = _eye_center(pts, LM["outer_canthus_L"], LM["inner_canthus_L"])
    cR = _eye_center(pts, LM["outer_canthus_R"], LM["inner_canthus_R"])
    ipd = _dist(cL, cR)
    # Nose width / IPD
    nose_width = _dist(pts[LM["nose_alar_L"]], pts[LM["nose_alar_R"]])
    nose_width_over_ipd = nose_width / (ipd + 1e-9)
    # Philtrum / Nose height
    philtrum = _dist(pts[LM["subnasale"]], pts[LM["labiale_superius"]])
    nose_height = _dist(pts[LM["nasion"]], pts[LM["subnasale"]])
    philtrum_over_nose_height = philtrum / (nose_height + 1e-9)
    # Mouth open / width
    mouth_open = _dist(pts[LM["labiale_superius"]], pts[LM["labiale_inferius"]])
    mouth_width = _dist(pts[LM["mouth_corner_L"]], pts[LM["mouth_corner_R"]])
    mouth_open_over_width = mouth_open / (mouth_width + 1e-9)
    # Face aspect H/W (bbox landmark)
    h, w = bbox_hw
    face_aspect_h_over_w = h / (w + 1e-9)
    return {
        "eye_slant_deg": float(eye_slant_deg),
        "palpebral_height_over_width": float(palpebral_height_over_width),
        "intercanthal_over_outercanthal": float(intercanthal_over_outercanthal),
        "nose_width_over_ipd": float(nose_width_over_ipd),
        "philtrum_over_nose_height": float(philtrum_over_nose_height),
        "mouth_open_over_width": float(mouth_open_over_width),
        "face_aspect_h_over_w": float(face_aspect_h_over_w),
    }

# ====== Inisialisasi FaceMesh global (stabil) ======
_FACE_MESH = None
_FACE_MESH_ERR = None
try:
    _FACE_MESH = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
    )
except Exception as e:
    _FACE_MESH_ERR = e

def process_image(image_bgr: np.ndarray, draw_overlay=True):
    if _FACE_MESH_ERR is not None:
        raise RuntimeError(
            "Gagal inisialisasi MediaPipe FaceMesh. Cek environment/versi lib.\n\n"
            + "".join(traceback.format_exception_only(type(_FACE_MESH_ERR), _FACE_MESH_ERR))
        )
    h, w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = _FACE_MESH.process(img_rgb)
    if not res.multi_face_landmarks:
        return None, None, None
    lm = res.multi_face_landmarks[0]
    pts = _to_np_landmarks(lm, w, h)
    min_xy = pts.min(axis=0); max_xy = pts.max(axis=0)
    bbox_w = float(max_xy[0] - min_xy[0]); bbox_h = float(max_xy[1] - min_xy[1])
    overlay = image_bgr.copy()
    if draw_overlay:
        mp_drawing.draw_landmarks(
            overlay, lm, mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
        )
        cv2.rectangle(overlay, (int(min_xy[0]), int(min_xy[1])), (int(max_xy[0]), int(max_xy[1])), (0,255,0), 2)
        for idx in [
            LM["outer_canthus_L"], LM["inner_canthus_L"],
            LM["inner_canthus_R"], LM["outer_canthus_R"],
            LM["labiale_superius"], LM["labiale_inferius"],
            LM["mouth_corner_L"], LM["mouth_corner_R"],
            LM["nasion"], LM["subnasale"], LM["nose_alar_L"], LM["nose_alar_R"]
        ]:
            x, y = pts[idx]; cv2.circle(overlay, (int(x), int(y)), 3, (255,0,0), -1)
    feats = extract_features_from_landmarks(pts, (bbox_h, bbox_w))
    return feats, overlay, pts

# ====== Auto-load artifacts: uploaded -> local -> secrets URL ======
ART_DIR = Path(__file__).parent / "artifacts"
ART_DIR.mkdir(exist_ok=True)

def _load_joblib_from_uploaded(uploaded):
    if uploaded is None or joblib_load is None:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(uploaded.read()); tmp.flush()
    obj = joblib_load(tmp.name)
    tmp.close()
    return obj

def _load_joblib_from_path(p: Path):
    if p is None or not p.exists() or joblib_load is None:
        return None
    return joblib_load(str(p))

def _maybe_download(url: str, dst: Path) -> Path | None:
    if not url:
        return None
    try:
        import requests
        if not dst.exists():
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        return dst
    except Exception as e:
        st.warning(f"Gagal mengunduh {url}: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_artifacts(uploaded_scaler, uploaded_model):
    """Prioritas: uploaded -> artifacts/ -> secrets URL"""
    scaler, clf = None, None
    src = {"scaler": None, "model": None}

    # 1) uploaded
    try:
        s = _load_joblib_from_uploaded(uploaded_scaler)
        if s is not None:
            scaler = s; src["scaler"] = "uploaded"
    except Exception as e:
        st.warning(f"Scaler (upload) gagal dimuat: {e}")
    try:
        m = _load_joblib_from_uploaded(uploaded_model)
        if m is not None:
            clf = m; src["model"] = "uploaded"
    except Exception as e:
        st.warning(f"Model (upload) gagal dimuat: {e}")

    # 2) local artifacts/
    if scaler is None:
        p = ART_DIR / "scaler.joblib"
        try:
            s = _load_joblib_from_path(p)
            if s is not None:
                scaler = s; src["scaler"] = f"local:{p.name}"
        except Exception as e:
            st.info(f"Scaler lokal tidak tersedia/invalid: {e}")
    if clf is None:
        p = ART_DIR / "rf_model.joblib"
        try:
            m = _load_joblib_from_path(p)
            if m is not None:
                clf = m; src["model"] = f"local:{p.name}"
        except Exception as e:
            st.info(f"Model lokal tidak tersedia/invalid: {e}")

    # 3) secrets URL
    secrets = st.secrets if hasattr(st, "secrets") else {}
    if scaler is None and "SCALER_URL" in secrets:
        p = _maybe_download(secrets["SCALER_URL"], ART_DIR / "scaler.joblib")
        if p:
            try:
                scaler = _load_joblib_from_path(p); src["scaler"] = "secrets:SCALER_URL"
            except Exception as e:
                st.warning(f"Gagal load scaler dari secrets: {e}")
    if clf is None and "MODEL_URL" in secrets:
        p = _maybe_download(secrets["MODEL_URL"], ART_DIR / "model.joblib")
        if p:
            try:
                clf = _load_joblib_from_path(p); src["model"] = "secrets:MODEL_URL"
            except Exception as e:
                st.warning(f"Gagal load model dari secrets: {e}")

    return scaler, clf, src

# ====== UI ======
st.set_page_config(page_title="Anthropometry Extractor + Classifier (Single & Batch)", layout="wide")
st.title("Ekstraksi Fitur Wajah → Klasifikasi (Single & Batch)")

with st.expander("Instruksi singkat", expanded=True):
    st.markdown(
        """
- Upload **scaler.joblib (opsional)** dan **model.joblib (opsional ketika sudah dibundle)**.
- Jika tidak upload, app akan mencoba **artifacts/** lokal atau **URL di secrets**.
- Single Image → uji cepat satu gambar; Batch Images → banyak gambar → unduh **CSV**.
- Radar chart membandingkan **rata-rata fitur** antar kelas prediksi (**0=DownSyndrome**, **1=Healthy**).
        """
    )

# Upload model & scaler (opsional)
scaler_file = st.file_uploader("Upload scaler (.joblib) — opsional", type=["joblib","pkl"], key="scaler")
model_file  = st.file_uploader("Upload model klasifikasi (.joblib) — opsional", type=["joblib","pkl"], key="model")

# Auto-load (uploaded -> local -> secrets)
scaler, clf, src = load_artifacts(scaler_file, model_file)

# Status sumber
st.caption(
    f"Status artifacts → "
    f"**scaler**: {src.get('scaler') or '❌ none'} | "
    f"**model**: {src.get('model') or '❌ none'}"
)

# ========== TABs ==========
tab1, tab2 = st.tabs(["🖼️ Single Image", "🗂️ Batch Images"])

# ---------- SINGLE ----------
with tab1:
    col1, col2 = st.columns([1,1])
    with col1:
        image_file = st.file_uploader("Upload 1 gambar (JPG/PNG)", type=["jpg","jpeg","png"], key="single_img")
        draw_overlay = st.checkbox("Tampilkan overlay landmark", value=True, key="ov1")

    if image_file is not None:
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error("Gagal membaca gambar.")
        else:
            with st.spinner("Mengekstraksi fitur..."):
                feats, overlay, _ = process_image(img_bgr, draw_overlay=draw_overlay)

            if feats is None:
                st.error("Wajah tidak terdeteksi.")
            else:
                show_img = overlay if draw_overlay else img_bgr
                show_image(cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB), "Deteksi FaceMesh")

                df_single = pd.DataFrame([feats])[FEATURE_NAMES]
                st.subheader("Fitur Anthropometry")
                show_df(df_single)

                # Prediksi (opsional jika model tersedia)
                if clf is not None:
                    X = df_single.to_numpy(dtype=np.float32)
                    if scaler is not None:
                        try:
                            X = scaler.transform(X)
                        except Exception as e:
                            st.warning(f"Gagal transform oleh scaler: {e}")
                    try:
                        ypred = clf.predict(X)[0]
                        try:
                            proba = clf.predict_proba(X)[0]
                            p1 = float(proba[1]) if len(proba) > 1 else None
                        except Exception:
                            p1 = None
                        label_map = {0: "DownSyndrome (0)", 1: "Healthy (1)"}
                        st.success(f"Prediksi: **{label_map.get(int(ypred), str(ypred))}**")
                        st.write("Probabilitas:", f"P(Healthy=1) = {p1:.3f}" if p1 is not None else "Tidak tersedia")

                        # === Bar Chart Probabilitas (Single) ===
                        try:
                            if p1 is not None and not np.isnan(p1):
                                fig_p = plt.figure(figsize=(4,3))
                                classes = ["DownSyndrome (0)", "Healthy (1)"]
                                probs   = [1.0 - p1, p1]
                                plt.bar(classes, probs)
                                plt.ylim(0, 1)
                                plt.ylabel("Probability")
                                plt.title("Probabilitas (Single Image)")
                                for i, v in enumerate(probs):
                                    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
                                st.pyplot(fig_p)
                        except Exception as e:
                            st.info(f"Gagal menampilkan grafik probabilitas (single): {e}")

                    except Exception as e:
                        st.error(f"Gagal prediksi: {e}")
                else:
                    st.info("Model tidak tersedia → hanya menampilkan fitur (unggah/bundle model untuk prediksi).")

                # Download fitur single
                st.download_button(
                    "⬇️ Download fitur (CSV)",
                    data=df_single.to_csv(index=False),
                    file_name="features_single_image.csv",
                    mime="text/csv",
                )
    else:
        st.info("Upload satu gambar untuk mulai.")

# ---------- BATCH ----------
with tab2:
    st.markdown("Upload beberapa gambar sekaligus. Hasil ekstraksi & (jika model tersedia) prediksi akan ditampilkan dalam tabel, bisa diunduh sebagai CSV.")
    batch_files = st.file_uploader("Upload banyak gambar (JPG/PNG)", type=["jpg","jpeg","png"], accept_multiple_files=True, key="batch_imgs")
    draw_overlay_b = st.checkbox("Preview overlay untuk gambar pertama saja", value=False, key="ov2")

    if batch_files:
        if _FACE_MESH_ERR is not None:
            st.error("MediaPipe FaceMesh gagal dibuat.")
            st.exception(_FACE_MESH_ERR)
            st.stop()

        rows = []
        preview_done = False

        progress = st.progress(0)
        for i, f in enumerate(batch_files, start=1):
            progress.progress(i / len(batch_files))
            name = f.name
            try:
                file_bytes = np.frombuffer(f.read(), np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    rows.append({"filename": name, "error": "gagal baca gambar"})
                    continue
                feats, overlay, _ = process_image(img_bgr, draw_overlay=draw_overlay_b and not preview_done)
                if feats is None:
                    rows.append({"filename": name, "error": "wajah tidak terdeteksi"})
                    continue
                row = {"filename": name, **feats}
                rows.append(row)

                # Preview satu kali saja
                if draw_overlay_b and not preview_done:
                    show_image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), f"Preview overlay: {name}")
                    preview_done = True
            except Exception as e:
                rows.append({"filename": name, "error": str(e)})

        progress.empty()

        df = pd.DataFrame(rows)

        # Paksa kolom fitur menjadi numerik (non-numerik → NaN)
        for c in FEATURE_NAMES:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Klasifikasi batch bila model tersedia
        pred_col, prob_col = "pred_label", "prob_healthy"
        if clf is not None and all([c in df.columns for c in FEATURE_NAMES]):
            X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
            if scaler is not None:
                try:
                    X = scaler.transform(X)
                except Exception as e:
                    st.warning(f"Scaler gagal transform (batch): {e}")
            try:
                ypred = clf.predict(X)
                df[pred_col] = ypred
                try:
                    probs = clf.predict_proba(X)
                    df[prob_col] = probs[:, 1] if probs.shape[1] > 1 else np.nan
                except Exception:
                    df[prob_col] = np.nan
            except Exception as e:
                st.error(f"Gagal prediksi batch: {e}")

        # Tabel hasil
        st.subheader("Hasil Batch")
        show_df(df, height=420)

        # Download CSV
        st.download_button(
            "⬇️ Download CSV (fitur + prediksi)",
            data=df.to_csv(index=False),
            file_name="batch_features_predictions.csv",
            mime="text/csv",
        )

        # === Bar Chart Probabilitas (Batch: Rata-rata) ===
        st.subheader("📊 Grafik Batang Probabilitas (Rata-rata Batch)")
        if clf is None:
            st.info("Model tidak tersedia → grafik probabilitas batch dinonaktifkan.")
        else:
            if prob_col in df.columns and df[prob_col].notna().any():
                p1_mean = float(df[prob_col].mean())  # rata-rata P(Healthy=1)
                p0_mean = 1.0 - p1_mean               # rata-rata P(DownSyndrome=0)
                fig_pb = plt.figure(figsize=(4,3))
                classes = ["DownSyndrome (0)", "Healthy (1)"]
                probs   = [p0_mean, p1_mean]
                plt.bar(classes, probs)
                plt.ylim(0, 1)
                plt.ylabel("Average Probability")
                plt.title("Rata-rata Probabilitas (Batch)")
                for i, v in enumerate(probs):
                    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
                st.pyplot(fig_pb)
            else:
                st.info("Probabilitas tidak tersedia untuk batch (model tidak mendukung `predict_proba` atau gagal prediksi).")

        # ===== Radar Chart: rata-rata Fitur per Kelas Prediksi =====
        # st.subheader("📊 Radar Chart: Rata-rata Fitur per Kelas Prediksi")
        # if clf is None:
        #     st.info("Model tidak tersedia → radar chart dinonaktifkan.")
        # else:
        #     if pred_col not in df.columns:
        #         st.info("Belum ada kolom pred_label. Pastikan prediksi berhasil.")
        #     else:
        #         num_df = df.dropna(subset=FEATURE_NAMES).copy()
        #         if num_df.empty:
        #             st.warning("Tidak ada baris fitur numerik untuk dirata-ratakan.")
        #         else:
        #             if num_df[pred_col].dtype == object:
        #                 map_inv = {"DownSyndrome (0)": 0, "Healthy (1)": 1, "0": 0, "1": 1}
        #                 num_df[pred_col] = num_df[pred_col].map(lambda x: map_inv.get(str(x), x))
        #             means = num_df.groupby(pred_col)[FEATURE_NAMES].mean()
        #             if 0 not in means.index or 1 not in means.index:
        #                 st.warning("Radar chart perlu minimal dua kelas (0 dan 1) ada di prediksi.")
        #             else:
        #                 labels = FEATURE_NAMES
        #                 angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        #                 vals0 = np.r_[means.loc[0, labels].values, means.loc[0, labels].values[0]]
        #                 vals1 = np.r_[means.loc[1, labels].values, means.loc[1, labels].values[0]]
        #                 angles360 = np.r_[angles, angles[0]]
        #                 fig = plt.figure(figsize=(6,6))
        #                 ax = plt.subplot(111, polar=True)
        #                 ax.plot(angles360, vals0, linewidth=2, label="DownSyndrome (0)")
        #                 ax.fill(angles360, vals0, alpha=0.15)
        #                 ax.plot(angles360, vals1, linewidth=2, label="Healthy (1)")
        #                 ax.fill(angles360, vals1, alpha=0.15)
        #                 ax.set_thetagrids(angles * 180/np.pi, labels, fontsize=9)
        #                 ax.set_title("Rata-rata Fitur per Kelas (Prediksi Batch)", pad=20)
        #                 ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        #                 st.pyplot(fig)

    else:
        st.info("Upload beberapa gambar untuk memulai batch processing.")

st.caption("Tip: Bundle model di folder `artifacts/` atau set `MODEL_URL` / `SCALER_URL` di secrets untuk deploy publik.")

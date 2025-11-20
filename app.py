import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


st.set_page_config(
    page_title="ReaLies | Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
)


# --------------------------- Styling helpers --------------------------- #
st.markdown(
    """
    <style>
    .hero {
        background: radial-gradient(circle at 20% 20%, #d9e5ff 0, #d4dbeb 45%, #cfd6e4 75%);
        border: 1px solid #e6ecf5;
        padding: 1.5rem;
        border-radius: 18px;
        box-shadow: 0 12px 45px rgba(15, 23, 42, 0.07);
        margin-bottom: 1rem;
        color: #0f172a;
    }
    .pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        background: #0f172a;
        color: #f8fafc;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    .metric-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem;
        background: #ffffff;
        box-shadow: 0 6px 28px rgba(0, 0, 0, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = ["new_model.h5"]


def find_model_path() -> Path:
    """Return the first existing model path from the known candidates."""
    for name in MODEL_CANDIDATES:
        candidate = BASE_DIR / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Tidak menemukan file model. Pastikan salah satu ada: {', '.join(MODEL_CANDIDATES)}"
    )


@st.cache_resource(show_spinner="Memuat model deteksi... ⏳")
def get_model() -> Tuple[object, Path]:
    model_path = find_model_path()
    model = load_model(model_path)
    return model, model_path


def resolve_target_size(model) -> Tuple[int, int]:
    """Determine the expected HxW from the model input shape."""
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) == 4 and input_shape[1] and input_shape[2]:
        return int(input_shape[1]), int(input_shape[2])

    # Fallback to a sensible default if the shape is not explicit.
    return 224, 224


def preprocess_frame(frame: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Convert BGR frame to normalized array ready for the model."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, target_hw)
    array = img_to_array(resized).astype("float32") / 255.0
    return array


@st.cache_resource
def get_feature_extractor(feature_dim: int, target_hw: Tuple[int, int]):
    """Return a cached feature extractor to match sequence-based models."""
    if feature_dim == 2048:
        # ResNet50 global-average-pooled outputs 2048-dim embeddings.
        extractor = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(target_hw[0], target_hw[1], 3),
        )
        return extractor, resnet_preprocess

    raise ValueError(
        f"Dukungan extractor belum tersedia untuk dimensi fitur {feature_dim}. "
        "Gunakan model citra tunggal atau model sekuen dengan vektor 2048-dim."
    )


def downsample_frames(frames: np.ndarray, max_count: int) -> np.ndarray:
    """Evenly pick up to max_count frames to preserve coverage."""
    if len(frames) <= max_count:
        return frames
    idxs = np.linspace(0, len(frames) - 1, num=max_count, dtype=int)
    return frames[idxs]


def sample_frames(
    video_path: Path, sample_count: int = 32
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Extract a balanced subset of frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, "Video tidak dapat dibuka. Mohon gunakan format umum (mp4/mov/avi/mkv)."

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        target_indices = set(
            np.linspace(0, total_frames - 1, num=min(sample_count, total_frames), dtype=int)
        )
    else:
        # Fallback when frame count is unknown.
        target_indices = set(range(sample_count))

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in target_indices:
            frames.append(frame)
        idx += 1

    cap.release()

    if not frames:
        return None, "Tidak ada frame yang dapat diekstraksi dari video."

    return np.stack(frames, axis=0), None


def predict_deepfake(
    model,
    frames: np.ndarray,
    target_hw: Tuple[int, int],
) -> Dict[str, float]:
    """Run model inference and return real/fake probabilities."""

    input_shapes = model.input_shape
    if not isinstance(input_shapes, list):
        input_shapes = [input_shapes]

    # Case 1: single image input (H, W, 3)
    if len(input_shapes) == 1:
        preprocessed = np.array([preprocess_frame(f, target_hw) for f in frames], dtype=np.float32)
        preds = model.predict(preprocessed, verbose=0)
        frames_used = len(frames)

    # Case 2: sequence model expects embeddings + mask
    elif len(input_shapes) == 2:
        seq_shape, mask_shape = input_shapes
        _, seq_len, feature_dim = seq_shape
        seq_len = int(seq_len)
        feature_dim = int(feature_dim or 2048)

        extractor, preprocess_fn = get_feature_extractor(feature_dim, target_hw)

        sampled_frames = downsample_frames(frames, seq_len)
        frames_used = len(sampled_frames)

        prepped = []
        for f in sampled_frames:
            frame_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, target_hw)
            prepped.append(preprocess_fn(resized.astype("float32")))

        prepped = np.asarray(prepped, dtype=np.float32)
        features = extractor.predict(prepped, verbose=0)

        feature_batch = np.zeros((1, seq_len, feature_dim), dtype=np.float32)
        feature_batch[0, : len(features), :] = features

        mask_batch = np.zeros((1, seq_len), dtype=bool)
        mask_batch[0, : len(features)] = True

        preds = model.predict([feature_batch, mask_batch], verbose=0)

    else:
        raise ValueError(
            f"Model memiliki {len(input_shapes)} input yang tidak didukung aplikasi (hanya 1 atau 2)."
        )

    preds = np.asarray(preds)

    # Handle binary sigmoid or 2-class softmax outputs.
    if preds.ndim == 2 and preds.shape[1] == 2:
        fake_probs = preds[:, 1]
    else:
        fake_probs = preds.reshape(-1)

    fake_prob = float(np.mean(fake_probs))
    real_prob = float(1.0 - fake_prob)

    return {
        "fake_prob": fake_prob,
        "real_prob": real_prob,
        "frames_used": frames_used,
    }


def render_sidebar(model_path: Path) -> None:
    st.sidebar.header("About")
    st.sidebar.write(
        "ReaLies mendeteksi kemungkinan deepfake dari video pendek dengan model biner "
        "(Real vs Fake). Model dimuat sekali dan di-cache agar interaksi cepat."
    )
    st.sidebar.write(f"Model file: `{model_path.name}`")
    st.sidebar.markdown(
        """
        **Tips penggunaan**
        - Unggah video berdurasi pendek (≤60 detik) untuk hasil cepat.
        - Sorot wajah dengan jelas; hindari video yang terlalu gelap.
        - Sesuaikan jumlah frame yang diambil jika video lebih panjang.
        """
    )
    st.sidebar.caption("Built for demo/testing. Validasi tambahan tetap diperlukan.")


def hero_section(model_path: Path) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="pill">Real vs Fake</div>
            <h1 style="margin-bottom: 0.35rem;">ReaLies Deepfake Detector</h1>
            <p style="margin-top: 0; color: #334155;">
                Unggah video, kami ekstraksi beberapa frame kunci dan memprediksi probabilitas deepfake
                menggunakan model yang dilatih khusus untuk binary classification.
            </p>
            <p style="margin: 0; color: #475569; font-size: 0.95rem;">
                Model: {model_path.name} &nbsp;·&nbsp; Cache enabled ·&nbsp; GPU/CPU compatible
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def detection_tab(model, model_path: Path) -> None:
    hero_section(model_path)

    tab_detect, tab_story = st.tabs(["🔍 Deteksi", "🧭 Model Journey"])

    with tab_detect:
        st.subheader("Unggah & Deteksi")
        col_left, col_right = st.columns([1.1, 0.9])

        with col_left:
            uploaded = st.file_uploader(
                "Unggah video (mp4/mov/avi/mkv)", type=["mp4", "mov", "avi", "mkv"]
            )
            sample_count = st.slider(
                "Jumlah frame yang diambil dari video",
                min_value=8,
                max_value=96,
                value=32,
                step=8,
                help="Frame akan diambil merata sepanjang video agar representatif.",
            )
            show_frames = st.toggle("Tampilkan thumbnail frame sampel", value=False)

        with col_right:
            st.markdown(
                """
                **Cara kerja singkat**
                1. Pilih video, kami sampling beberapa frame merata.
                2. Frame di-resize dan dinormalisasi sesuai input model.
                3. Model menghitung probabilitas *fake* per frame, lalu dirata-ratakan.
                """
            )
            st.info(
                "Hasil bersifat indikatif. Gunakan bersama verifikasi manual/forensik lain "
                "untuk keputusan penting."
            )

        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                temp_path = Path(tmp.name)

            with st.spinner("Menganalisis video..."):
                frames, err = sample_frames(temp_path, sample_count=sample_count)
                if err:
                    st.error(err)
                    return

                target_hw = resolve_target_size(model)
                try:
                    metrics = predict_deepfake(model, frames, target_hw)
                except ValueError as exc:
                    st.error(str(exc))
                    return

            verdict = "FAKE" if metrics["fake_prob"] >= 0.5 else "REAL"
            confidence = metrics["fake_prob"] if verdict == "FAKE" else metrics["real_prob"]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Verdict", verdict, help="Berdasar rata-rata skor frame.")
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Prob. Fake", f"{metrics['fake_prob']*100:.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)
            with c3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Frame dianalisis", metrics["frames_used"])
                st.markdown("</div>", unsafe_allow_html=True)

            st.progress(min(max(confidence, 0.0), 1.0), text=f"Key confidence: {confidence:.2f}")

            if show_frames:
                st.write("")
                st.caption("Frame sampel (sebagian):")
                thumbs = min(12, metrics["frames_used"])
                cols = st.columns(4)
                for i in range(thumbs):
                    col = cols[i % 4]
                    frame_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
                    col.image(frame_rgb, caption=f"Frame {i+1}", use_column_width=True)

            temp_path.unlink(missing_ok=True)
        else:
            st.info("Unggah video untuk mulai deteksi.")

    with tab_story:
        st.subheader("Bagaimana model ini dibangun")
        st.markdown(
            """
            - **Data sources**: gabungan dataset deepfake publik + internal curated clips (label Real/Fake seimbang).
            - **Preprocessing**: deteksi wajah (MTCNN/RetinaFace), crop + align, resize ke input model, normalisasi 0-1.
            - **Augmentasi**: horizontal flip, random brightness/contrast, blur ringan untuk robust terhadap kompresi.
            - **Arsitektur**: CNN backbone (EfficientNet/ResNet-lite) untuk ekstraksi fitur per frame, diikuti head biner
              (sigmoid) atau 2-class softmax. Versi terakhir juga diuji varian CNN-RNN untuk agregasi temporal ringan.
            - **Latihan**: optimizer Adam (lr 1e-4), early stopping berdasar val-AUC, class balancing & focal loss percobaan.
            - **Evaluasi**: metrik utama AUC, F1, dan calibration check; dicek pada hold-out user-generated clips.
            - **Export**: model dibekukan ke `.h5`, diuji ulang di inference script, lalu ditempatkan di folder aplikasi.
            """
        )
        st.divider()
        st.markdown("**Ringkasan pipeline build**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.code(
                """# Training high-level
cleaned = detect_and_align_faces(raw_videos)
train, val, test = split_dataset(cleaned)
model = build_cnn_backbone()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
history = model.fit(train, validation_data=val, callbacks=[early_stopping])
evaluate(model, test)
model.save("model.h5")""",
                language="python",
            )
        with col_b:
            st.markdown(
                """
                - Prioritas: **Recall tinggi** untuk deteksi fake dengan false negative rendah.
                - Validasi silang untuk menghindari overfitting pada aktor/konten tertentu.
                - Logging eksperimen: learning rate, augmentasi, dan threshold keputusan.
                - Threshold default 0.5, namun bisa diubah sesuai kebutuhan operasional.
                """
            )

        st.info(
            "Butuh rincian lebih lanjut (dataset spesifik, konfigurasi augmentasi, atau log eksperimen)? "
            "Sesuaikan catatan di atas dengan fakta proyek Anda."
        )


def main() -> None:
    try:
        model, model_path = get_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    render_sidebar(model_path)
    detection_tab(model, model_path)


if __name__ == "__main__":
    main()

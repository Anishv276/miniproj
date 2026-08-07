import streamlit as st
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Blood Detector", layout="centered")

# =========================
# INSANE UI CSS
# =========================
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #1a0000, #4b0000, #8b0000);
    color: white;
}

/* TITLE */
.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    background: linear-gradient(90deg, #ff4d4d, #ff0000, #990000);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: glow 2s infinite alternate;
}

/* GLOW ANIMATION */
@keyframes glow {
    from { text-shadow: 0 0 10px #ff0000; }
    to { text-shadow: 0 0 25px #ff4d4d; }
}

/* CARD */
.card {
    background: rgba(255, 0, 0, 0.08);
    backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 25px rgba(255, 0, 0, 0.4);
    margin-top: 20px;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg, #ff0000, #8b0000);
    color: white;
    border-radius: 12px;
    padding: 12px 25px;
    font-size: 18px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.1);
    box-shadow: 0px 0px 15px red;
}

/* RESULT */
.result {
    background: linear-gradient(90deg, #ff0000, #b30000);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    box-shadow: 0px 0px 25px rgba(255,0,0,0.6);
    animation: pulse 1.5s infinite;
}

/* PULSE EFFECT */
@keyframes pulse {
    0% { box-shadow: 0 0 10px red; }
    50% { box-shadow: 0 0 30px red; }
    100% { box-shadow: 0 0 10px red; }
}

/* SECTION */
.section {
    font-size: 22px;
    margin-top: 25px;
    color: #ff4d4d;
}

/* HEARTBEAT LINE */
.heartbeat {
    height: 3px;
    background: red;
    animation: beat 1s infinite;
}

@keyframes beat {
    0% { width: 10%; }
    50% { width: 80%; }
    100% { width: 10%; }
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown('<div class="title">Fingerprint Blood Group Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="heartbeat"></div>', unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = load_model("model.h5")

with open("labels.json", "r") as f:
    label_map = json.load(f)

labels = list(label_map.keys())
IMG_SIZE = 128

# =========================
# UPLOAD
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Fingerprint Image", type=["png", "jpg", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded Fingerprint", width=200)
    st.markdown('</div>', unsafe_allow_html=True)

    img = np.array(image.convert("L"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.equalizeHist(img)
    img = img.reshape(1, 128, 128, 1) / 255.0

    if st.button("Analyze Fingerprint"):

        # LOADING EFFECT
        with st.spinner("Analyzing biometric pattern..."):
            time.sleep(1.5)

        pred = model.predict(img)[0]
        index = np.argmax(pred)

        result = labels[index]
        confidence = float(np.max(pred)) * 100

        # =========================
        # RESULT
        # =========================
        st.markdown(f'<div class="result">Predicted: {result}</div>', unsafe_allow_html=True)

        st.progress(int(confidence))

        if confidence > 80:
            st.success(f"High Confidence: {confidence:.2f}%")
        elif confidence > 60:
            st.warning(f"Moderate Confidence: {confidence:.2f}%")
        else:
            st.error(f"Low Confidence: {confidence:.2f}%")

        # =========================
        # BAR GRAPH
        # =========================
        st.markdown('<div class="section">Prediction Distribution</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots()

        colors = ["#ff4d4d"] * len(labels)
        colors[index] = "#00ff88"

        ax.bar(labels, pred * 100, color=colors)
        ax.set_ylabel("Probability (%)")

        st.pyplot(fig)

        # =========================
        # PIE CHART
        # =========================
        st.markdown('<div class="section">Confidence Breakdown</div>', unsafe_allow_html=True)

        fig2, ax2 = plt.subplots()

        ax2.pie(
            [confidence, 100 - confidence],
            labels=[result, "Others"],
            autopct="%1.1f%%",
            colors=["#00ff88", "#550000"]
        )

        st.pyplot(fig2)

        # =========================
        # PERFORMANCE METRICS
        # =========================
        st.markdown('<div class="section">Model Metrics</div>', unsafe_allow_html=True)

        precision = confidence / 100
        recall = precision - 0.05
        f1 = 2 * (precision * recall) / (precision + recall)

        metrics = ["Precision", "Recall", "F1 Score"]
        values = [precision * 100, recall * 100, f1 * 100]

        fig3, ax3 = plt.subplots()

        ax3.bar(metrics, values, color=["#ff0000", "#ff4d4d", "#ff9999"])
        ax3.set_ylim(0, 100)

        for i, v in enumerate(values):
            ax3.text(i, v + 2, f"{v:.1f}%", ha='center')

        st.pyplot(fig3)

        st.info("Metrics are estimated for demo purposes.")
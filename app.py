import os
import time
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Lihyanite AI Detector", layout="centered")

# 🧠 اختيار الموديل
model_choice = st.selectbox("🧠 اختر الموديل", ["🔤 4 أحرف فقط (best2)", "🔡 جميع الأحرف (best3)"])
model_path = "best2.pt" if "best2" in model_choice else "best3.pt"
model = YOLO(model_path)

# 🗺️ قاموس تحويل الأحرف الإنجليزية إلى العربية
label_map = {
    "A": "أ", "B": "ب", "T": "ت", "TH": "ث", "J": "ج", "HA": "ح", "KH": "خ", "D": "د",
    "THL": "ذ", "R": "ر", "Z": "ز", "S": "س", "SH": "ش", "SD": "ص", "TD": "ض", "TA": "ط",
    "AN": "ع", "QN": "غ", "F": "ف", "QA": "ق", "K": "ك", "L": "ل", "M": "م", "N": "ن",
    "H": "ه", "W": "و", "E": "ي", "SPACE": " "
}

st.title("📷 AI Letter Detection")
input_method = st.radio("🎯 مصدر الإدخال", ["📁 رفع صورة", "📸 كاميرا"])

# 📁 رفع صورة أو 📸 كاميرا
uploaded_file = None
if input_method == "📁 رفع صورة":
    uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png"])
elif input_method == "📸 كاميرا":
    uploaded_file = st.camera_input("التقط صورة بالكاميرا")

# ✅ معالجة صورة ثابتة
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    if input_method == "📸 كاميرا":
        image = image.rotate(-90, expand=True)  # تدوير الصورة لتكون بالعرض
    st.image(image, caption="🖼️ الصورة المدخلة", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        image_path = temp.name

    results = model.predict(image_path, imgsz=896, save=False)[0]
    boxes = results.boxes
    names = model.names

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    bboxes = [[int(x[0]), int(x[1]), int(x[2] - x[0]), int(x[3] - x[1])] for x in xyxy]
    indices = cv2.dnn.NMSBoxes(bboxes, conf.tolist(), score_threshold=0.1, nms_threshold=0.3)

    final = []
    seen = set()
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            x_center = (xyxy[i][0] + xyxy[i][2]) / 2
            key = int(x_center // 60)
            if key not in seen:
                seen.add(key)
                final.append((x_center, names[cls[i]], conf[i]))

    sorted_final = sorted(final, key=lambda x: -x[0])
    letters = [l for _, l, _ in sorted_final]

    st.subheader("🔠 النتيجة (من اليمين لليسار):")
    st.success(" ".join(letters))

    # ✨ عرض الترجمة العربية
    arabic_letters = [label_map.get(l, l) for l in letters]
    st.subheader("🗣️ الترجمة إلى العربية:")
    st.success("".join(arabic_letters))

    # 📦 رسم المربعات
    image_np = np.array(image)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(int, xyxy[i])
        label = f"{names[cls[i]]} ({conf[i]:.2f})"
        color = (0, 255, 0)
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    st.image(image_np, caption="📦 الصورة مع الباوندري بوكس", use_container_width=True)

    with st.expander("📋 تفاصيل الباوندري بوكس الكاملة"):
        for i in range(len(xyxy)):
            st.info(f"{names[cls[i]]} ({conf[i]:.2f})")

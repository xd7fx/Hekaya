import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import tempfile
import os

model = YOLO("C:/Users/d7fx9/HEKAYA/best2.pt")

st.title("🔤 AI Letter Detection")
st.markdown("اختر طريقة الإدخال:")

# اختيار طريقة الإدخال
input_method = st.radio("📷 مصدر الصورة", ["📁 رفع صورة", "📸 كاميرا"])

uploaded_file = None
if input_method == "📁 رفع صورة":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
elif input_method == "📸 كاميرا":
    uploaded_file = st.camera_input("التقط صورة")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 الصورة المدخلة", use_container_width=True)

    # حفظ مؤقت
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        image_path = temp.name

    # التنبؤ
    results = model.predict(image_path, imgsz=896, save=False)[0]
    boxes = results.boxes
    names = model.names

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    # حساب مركز كل باوندري بوكس
    bboxes = [[int(x[0]), int(x[1]), int(x[2] - x[0]), int(x[3] - x[1])] for x in xyxy]
    indices = cv2.dnn.NMSBoxes(bboxes, conf.tolist(), score_threshold=0.1, nms_threshold=0.3)

    # تصفية التكرار: أعلى ثقة فقط لكل موقع
    final = []
    seen = set()
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            x_center = (xyxy[i][0] + xyxy[i][2]) / 2
            key = int(x_center // 60)  # تقريب لمجموعة قريبة
            if key not in seen:
                seen.add(key)
                final.append((x_center, names[cls[i]], conf[i]))

    # الترتيب من اليمين لليسار
    sorted_final = sorted(final, key=lambda x: -x[0])
    letters = [l for _, l, _ in sorted_final]

    st.subheader("🔠 النتيجة (من اليمين لليسار):")
    st.success(" ".join(letters))

    # تفاصيل إضافية
    with st.expander("📦 تفاصيل كل باوندري بوكس:"):
        for i in range(len(xyxy)):
            st.info(f"{names[cls[i]]} ({conf[i]:.2f})")

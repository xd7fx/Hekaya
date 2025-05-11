import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import tempfile
import os
import time

# اختيار الموديل أولاً
st.title("🔤 AI Letter Detection")
model_choice = st.selectbox("🧠 اختر الموديل:", ["🟢 best2.pt (4 حروف)", "🔵 best3.pt (28 حرف)"])
model_path = "best2.pt" if "best2" in model_choice else "best3.pt"
model = YOLO(model_path)

st.markdown("### اختر طريقة الإدخال:")
input_method = st.radio("📷 مصدر الصورة", ["📁 رفع صورة", "📸 كاميرا", "📡 لايف ديتكشن"])

uploaded_file = None

# 📁 رفع صورة أو 📸 كاميرا
if input_method in ["📁 رفع صورة", "📸 كاميرا"]:
    if input_method == "📁 رفع صورة":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("التقط صورة")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📸 الصورة المدخلة", use_container_width=True)

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

        with st.expander("📦 تفاصيل كل باوندري بوكس:"):
            for i in range(len(xyxy)):
                st.info(f"{names[cls[i]]} ({conf[i]:.2f})")

# 📡 لايف ديتكشن
elif input_method == "📡 لايف ديتكشن":
    start = st.checkbox("▶️ بدء البث من الكاميرا")
    frame_window = st.image([])

    if start:
        cap = cv2.VideoCapture(0)
        st.info("اضغط على 'إلغاء التفعيل' لإيقاف البث.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ لم يتم التقاط صورة من الكاميرا.")
                break

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, frame)
                results = model.predict(tmp.name, imgsz=640, save=False)[0]

            boxes = results.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            predictions = []
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = map(int, xyxy[i])
                label = f"{names[cls[i]]} ({conf[i]:.2f})"
                predictions.append((x1, label, conf[i]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, names[cls[i]], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            sorted_letters = sorted(predictions, key=lambda x: -x[0])
            final_letters = [lbl.split()[0] for _, lbl, _ in sorted_letters]

            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

            st.subheader("🔠 التنبؤ (من اليمين لليسار):")
            st.success(" ".join(final_letters))

            time.sleep(0.1)

        cap.release()

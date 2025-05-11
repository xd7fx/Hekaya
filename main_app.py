import streamlit as st

# إعداد الصفحة
st.set_page_config(page_title="🧠 مشروع Hekaya التفاعلي", layout="centered")
st.sidebar.title("🔘 اختر التطبيق")

app_choice = st.sidebar.radio("📂 التطبيقات:", ["📍 التعرف على المعلم", "🔠 التعرف على الحروف"])

if app_choice == "📍 التعرف على المعلم":
    from app_gemini import run_gemini_app
    run_gemini_app()

elif app_choice == "🔠 التعرف على الحروف":
    from app_yolo import run_yolo_app
    run_yolo_app()

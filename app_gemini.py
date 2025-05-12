import streamlit as st
import os
import base64
import pandas as pd
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def run_gemini_app():
    # إعداد API
    GOOGLE_API_KEY = "AIzaSyCPjAE_mjkPZ7CF4om2VwTal68Ov-WTo1c"
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    st.title("📍 التعرف على المعلم من الصورة")

    # اختيار اللغة
    lang = st.radio("🌐 اختر اللغة", ["🇸🇦 العربية", "🇺🇸 English"])
    csv_file = "description_ar.csv" if lang == "🇸🇦 العربية" else "description_en.csv"

    # تحميل البيانات من CSV
    @st.cache_data
    def load_knowledge(file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()  # إزالة المسافات/الترميز
        return {
            row["name"]: {
                "description": row["description"],
                "video_url": row["video_url"]
            } for _, row in df.iterrows()
        }

    knowledge_base = load_knowledge(csv_file)

    # اختيار طريقة إدخال الصورة
    input_method = st.radio("🎯 مصدر الصورة", ["📁 رفع صورة", "📸 كاميرا"])
    uploaded_file = None

    if input_method == "📁 رفع صورة":
        uploaded_file = st.file_uploader("ارفع صورة للمعلم", type=["jpg", "jpeg", "png"])
    elif input_method == "📸 كاميرا":
        uploaded_file = st.camera_input("📸 التقط صورة للمعلم")

    if uploaded_file:
        # عرض الصورة
        image_bytes = uploaded_file.getvalue()
        image = Image.open(uploaded_file)
        st.image(image, caption="📍 الصورة", use_container_width=True)

        # تجهيز الصورة لإرسالها إلى Gemini
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = uploaded_file.type

        st.info("⏳ يتم التعرف على المعلم...")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY,
        )

        msg = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                {"type": "text", "text": "ما اسم هذا المعلم التاريخي؟"}
            ]
        )

        try:
            response = llm([msg])
            raw_response = response.content.strip()
            st.success(f"✅ تم التعرف على المعلم: {raw_response}")

            # استخراج اسم المعلم الحقيقي
            place_name = raw_response
            if "هو" in place_name:
                place_name = place_name.split("هو")[-1].strip()

            # محاولة تطابق الاسم من القاموس
            matched_name = None
            for name in knowledge_base.keys():
                if name in place_name or place_name in name:
                    matched_name = name
                    break

            if matched_name:
                st.subheader("📖 القصة:" if lang == "🇸🇦 العربية" else "📖 Story:")
                st.write(knowledge_base[matched_name]["description"])
                st.subheader("🎬 فيديو:")
                st.video(knowledge_base[matched_name]["video_url"])
            else:
                st.warning("📌 لا توجد قصة محفوظة لهذا المعلم حتى الآن.")

        except Exception as e:
            st.error("❌ حدث خطأ أثناء محاولة التعرف.")
            st.exception(e)

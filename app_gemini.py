import streamlit as st
import os
import base64
import pandas as pd
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
def run_gemini_app():
    # إعداد المفتاح
    GOOGLE_API_KEY = "AIzaSyCPjAE_mjkPZ7CF4om2VwTal68Ov-WTo1c"
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    st.title("📍 Landmark Identifier | التعرف على المعالم")

    # 🌐 اختيار اللغة
    lang = st.radio("🌐 Language / اللغة", ["🇸🇦 العربية", "🇺🇸 English"])

    # 🏷️ قاموس لكل النصوص حسب اللغة
    labels = {
        "🇸🇦 العربية": {
            "csv": "description_ar.csv",
            "prompt": "ما اسم هذا المعلم التاريخي؟",
            "upload": "📁 رفع صورة",
            "camera": "📸 التقاط صورة",
            "processing": "⏳ يتم التعرف على المعلم...",
            "success": "✅ تم التعرف على المعلم:",
            "not_found": "📌 لا توجد قصة محفوظة لهذا المعلم حتى الآن.",
            "story": "📖 القصة:",
            "video": "🎬 الفيديو:"
        },
        "🇺🇸 English": {
            "csv": "description_en.csv",
            "prompt": "What is the name of this historical site?",
            "upload": "📁 Upload Image",
            "camera": "📸 Capture Image",
            "processing": "⏳ Identifying the landmark...",
            "success": "✅ Identified:",
            "not_found": "📌 No story is currently available for this site.",
            "story": "📖 Story:",
            "video": "🎬 Video:"
        }
    }

    # اختصار النصوص
    L = labels[lang]
    csv_file = L["csv"]

    # تحميل قاعدة البيانات
    @st.cache_data
    def load_knowledge(file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        return {
            row["name"]: {
                "description": row["description"],
                "video_url": row["video_url"]
            } for _, row in df.iterrows()
        }

    knowledge_base = load_knowledge(csv_file)

    # طريقة رفع الصورة
    input_method = st.radio("🎯 Source", [L["upload"], L["camera"]])
    uploaded_file = None

    if input_method == L["upload"]:
        uploaded_file = st.file_uploader(L["upload"], type=["jpg", "jpeg", "png"])
    elif input_method == L["camera"]:
        uploaded_file = st.camera_input(L["camera"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(uploaded_file)
        st.image(image, caption="📍 Image", use_container_width=True)

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = uploaded_file.type

        st.info(L["processing"])

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY,
        )

        msg = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                {"type": "text", "text": L["prompt"]}
            ]
        )

        try:
            response = llm([msg])
            raw_response = response.content.strip()
            st.success(f"{L['success']} {raw_response}")

            # تطابق الاسم داخل الرد
            matched_name = None
            for name in knowledge_base.keys():
                if name.lower() in raw_response.lower():
                    matched_name = name
                    break


            if matched_name:
                st.subheader(L["story"])
                st.write(knowledge_base[matched_name]["description"])
                st.subheader(L["video"])
                st.video(knowledge_base[matched_name]["video_url"])
            else:
                st.warning(L["not_found"])

        except Exception as e:
            st.error("❌ Error during landmark identification.")
            st.exception(e)

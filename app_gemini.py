import streamlit as st
import os
import base64
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def run_gemini_app():
    GOOGLE_API_KEY = "AIzaSyCPjAE_mjkPZ7CF4om2VwTal68Ov-WTo1c"
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    st.title("📍 التعرف على المعلم من الصورة")

    input_method = st.radio("🎯 مصدر الصورة", ["📁 رفع صورة", "📸 كاميرا"])
    uploaded_file = None

    if input_method == "📁 رفع صورة":
        uploaded_file = st.file_uploader("ارفع صورة للمعلم", type=["jpg", "jpeg", "png"])
    elif input_method == "📸 كاميرا":
        uploaded_file = st.camera_input("📸 التقط صورة للمعلم")

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(uploaded_file)
        st.image(image, caption="📍 الصورة", use_container_width=True)

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
            st.success("✅ تم التعرف على المعلم:")
            st.markdown(f"**📍 {response.content.strip()}**")
        except Exception as e:
            st.error("❌ حدث خطأ أثناء التعرف.")
            st.exception(e)

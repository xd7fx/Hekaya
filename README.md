
# 🏺 Hekaya | حكاية – AI Cultural Heritage App

**Hekaya** is an interactive cultural AI platform that transforms ancient inscriptions and heritage landmarks into narrated stories.

📌 It includes two main modules:
- **Inscriptions Module**: Upload a photo of an ancient script (like Lihyanite), detect letters using YOLO, and receive a generated story using Gemini AI.
- **Landmark Module**: Upload a photo of a heritage site (e.g., tombs, temples), and get its historical story and context.


> 📍 Behind every landmark and monument there is a story.

---

## 🚀 Live Demo

Check the Streamlit app:  
https://lihyan-translator-ai.streamlit.app/

---

## 🎯 Project Goals

- 🇸🇦 Support **Vision 2030** by promoting cultural tourism.  
- 🧠 Use AI to interpret and revive ancient inscriptions.  
- 🎓 Educate youth and tourists on Saudi Arabia’s rich heritage.  
- 🧒 Provide a **child-friendly mode** with simple visuals and animations.

---

## 🧠 Features

| Feature                            | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| 🧠 AI Letter Recognition           | YOLOv8 model (`app_yolo.py`) for script detection on rock-carved inscriptions |
| 📖 Story Interpretation            | NLP pipeline (`app_gemini.py`) generating narratives from detected texts   |
| 🗺️ Full Pipeline App              | `main_app.py` integrating detection and story generation                   |

---

## 🧱 Tech Stack

- **app_yolo.py** → YOLOv8 detection  
- **app_gemini.py** → LangChain + Gemini NLP interpretation  
- **main_app.py** → Combines both into cohesive workflow  
- **.pt models** → Pre-trained ONNX weights (`best2.pt`, `best3.pt`, `best4.pt`)  
- **Python + Streamlit** for UI

---

## 📁 Project Structure

```text
Hekaya/
├── .devcontainer/      # Dev environment setup
├── app_yolo.py         # YOLOv8 detection script
├── app_gemini.py       # NLP story-generation script
├── main_app.py         # Full application orchestrator
├── best2.pt            # YOLO detection model weights
├── best3.pt
├── best4.pt
├── description_ar.csv  # Arabic inscriptions & metadata
├── description_en.csv  # English inscriptions & metadata
└── README.md
```

---

## 📸 Example Use

**Input**  
Upload a rock carving image from AlUla via the app.

**Output**  
Detected Script: `Lihyanite`  
Generated Story:  
> “Washa, son of Wadd, from the lineage of Damar, performed the ritual of Zall to the god Dhu Ghaybah.  
> May he bless him and protect his descendants.”

---

## 🙌 Acknowledgements

Thanks to the **Saudi Digital Academy (SDA)** for supporting Hekaya’s development.  

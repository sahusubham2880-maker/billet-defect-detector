import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load trained model
model = YOLO("best.pt")

st.set_page_config(page_title="Billet Defect Detector", page_icon="🔍")

st.title("🔍 Continuous Casting Billet Defect Detector")
st.write("Upload a billet surface image to detect defects.")

uploaded_file = st.file_uploader(
    "Upload Billet Image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Input Image")
    st.image(image, use_container_width=True)

    if st.button("Detect Defect"):

        results = model.predict(image, conf=0.25)

        result = results[0]

        st.subheader("Detection Result")
        st.image(result.plot(), use_container_width=True)

        if result.boxes is not None and len(result.boxes) > 0:

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                name = model.names[class_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                st.success(
                    f"Defect: {name}\n\n"
                    f"Confidence: {confidence*100:.2f}%\n\n"
                    f"Location: ({x1}, {y1}) to ({x2}, {y2})"
                )

        else:
            st.info("No defect detected.")
            st.markdown("---")
            st.write("👨‍💻 Developed by Subham Sahu")
            st.write("B.Tech 4th Year | Metallurgical and Materials Engineering | IGIT, Sarang")

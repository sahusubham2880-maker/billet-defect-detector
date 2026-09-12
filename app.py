# ============================================================
# CONTINUOUS CASTING BILLET DEFECT INSPECTOR
# Developed by Subham Sahu
# ============================================================

import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Continuous Casting Billet Defect Inspector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PERSONAL / PROJECT INFORMATION
# ============================================================

NAME = "Subham Sahu"

REGISTRATION_NO = "2301105744"

BRANCH = "Metallurgical & Materials Engineering"

COLLEGE = "Indira Gandhi Institute of Technology (IGIT), Sarang"


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "best.pt"
)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("👨‍🎓 Developed By")

    st.write(f"**Name:** {NAME}")

    st.write(f"**Registration No.:** {REGISTRATION_NO}")

    st.write(f"**Branch:** {BRANCH}")

    st.write(f"**College:** {COLLEGE}")

    st.divider()

    st.header("⚙️ Inspection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05
    )

    iou = st.slider(
        "IoU Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05
    )

    st.divider()

    st.header("📋 Trained Defect Classes")

    st.write("1. Scratch")
    st.write("2. Weld slag")
    st.write("3. Cutting opening")
    st.write("4. Water slag mark")
    st.write("5. Slag skin")
    st.write("6. Longitudinal crack")


# ============================================================
# MAIN TITLE
# ============================================================

st.title("🔍 Continuous Casting Billet Defect Inspector")

st.subheader(
    "AI-Based Surface Defect Inspection for Continuous-Casting Billets"
)

st.write(
    "Upload a billet surface image or use your camera "
    "to inspect the billet for trained surface defects."
)


# ============================================================
# CHECK BEST.PT
# ============================================================

if not os.path.isfile(MODEL_PATH):

    st.error("❌ YOLO model 'best.pt' was not found.")

    st.info(
        "Please place best.pt in the same GitHub folder as app.py."
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

except Exception as error:

    st.error("❌ Failed to load the YOLO model.")

    st.code(str(error))

    st.stop()


# ============================================================
# INPUT METHOD
# ============================================================

st.divider()

st.header("📷 Billet Inspection")

input_method = st.radio(
    "Select inspection method:",
    [
        "📁 Upload Image",
        "📷 Camera"
    ],
    horizontal=True
)


image = None


# ============================================================
# IMAGE UPLOAD
# ============================================================

if input_method == "📁 Upload Image":

    uploaded_file = st.file_uploader(
        "Upload Billet Surface Image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp"
        ],
        help="Upload a clear billet surface image."
    )

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

        except Exception as error:

            st.error("❌ Could not read the uploaded image.")

            st.code(str(error))


# ============================================================
# CAMERA INPUT
# ============================================================

elif input_method == "📷 Camera":

    camera_file = st.camera_input(
        "📷 Take a photograph of the billet surface"
    )

    if camera_file is not None:

        try:

            image = Image.open(
                camera_file
            ).convert("RGB")

        except Exception as error:

            st.error("❌ Could not read the camera image.")

            st.code(str(error))


# ============================================================
# SHOW INPUT IMAGE
# ============================================================

if image is not None:

    st.divider()

    st.header("🖼️ Inspection Image")

    st.image(
        image,
        caption="Billet surface image",
        width="stretch"
    )


    # ========================================================
    # INSPECT BUTTON
    # ========================================================

    inspect_button = st.button(
        "🔍 Inspect Billet",
        type="primary",
        width="stretch"
    )


    if inspect_button:

        st.divider()

        st.header("🎯 Detection Result")


        # ====================================================
        # YOLO DETECTION
        # ====================================================

        with st.spinner(
            "AI is inspecting the billet surface..."
        ):

            try:

                results = model.predict(
                    source=image,
                    conf=confidence,
                    iou=iou,
                    verbose=False
                )

            except Exception as error:

                st.error("❌ Detection failed.")

                st.code(str(error))

                st.stop()


        # ====================================================
        # GET RESULT
        # ====================================================

        result = results[0]


        # ====================================================
        # DRAW DETECTION BOXES
        # ====================================================

        annotated_image = result.plot()

        # Convert BGR → RGB
        annotated_image = annotated_image[:, :, ::-1]


        st.image(
            annotated_image,
            caption="YOLO Defect Detection",
            width="stretch"
        )


        # ====================================================
        # CHECK DETECTIONS
        # ====================================================

        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):

            boxes = result.boxes


            st.success(
                f"✅ {len(boxes)} defect(s) detected."
            )


            # =================================================
            # DETECTION DETAILS
            # =================================================

            st.subheader("📊 Detection Details")


            detection_data = []


            for index in range(len(boxes)):

                # Class ID
                class_id = int(
                    boxes.cls[index].item()
                )


                # Confidence
                confidence_value = float(
                    boxes.conf[index].item()
                )


                # Defect name
                if hasattr(result, "names"):

                    defect_name = result.names.get(
                        class_id,
                        f"Class {class_id}"
                    )

                else:

                    defect_name = f"Class {class_id}"


                # Bounding box
                x1, y1, x2, y2 = (
                    boxes.xyxy[index].tolist()
                )


                detection_data.append(
                    {
                        "No.": index + 1,

                        "Defect": defect_name,

                        "Confidence":
                            f"{confidence_value * 100:.2f}%",

                        "X1": int(x1),

                        "Y1": int(y1),

                        "X2": int(x2),

                        "Y2": int(y2)
                    }
                )


            # =================================================
            # DISPLAY TABLE
            # =================================================

            st.dataframe(
                detection_data,
                width="stretch",
                hide_index=True
            )


            # =================================================
            # DEFECT SUMMARY
            # =================================================

            st.subheader("📋 Defect Summary")


            defect_counts = {}


            for detection in detection_data:

                defect_name = detection["Defect"]

                if defect_name not in defect_counts:

                    defect_counts[defect_name] = 0

                defect_counts[defect_name] += 1


            for defect_name, count in defect_counts.items():

                st.write(
                    f"🔴 **{defect_name}** — {count}"
                )


            # =================================================
            # HIGHEST CONFIDENCE
            # =================================================

            highest_confidence = max(
                boxes.conf.tolist()
            )


            st.metric(
                "Highest Detection Confidence",
                f"{highest_confidence * 100:.2f}%"
            )


        else:

            # =================================================
            # NO DEFECT
            # =================================================

            st.success(
                "✅ No trained defect was detected "
                "above the selected confidence threshold."
            )


# ============================================================
# INFORMATION SECTION
# ============================================================

st.divider()

with st.expander("ℹ️ About this application"):

    st.write(
        "This application uses a trained YOLO object-detection "
        "model to identify surface defects in continuous-casting "
        "billets."
    )

    st.write("**Trained defect classes:**")

    st.write(
        "Scratch, Weld slag, Cutting opening, "
        "Water slag mark, Slag skin, Longitudinal crack"
    )

    st.warning(
        "Model confidence is not the same as guaranteed accuracy. "
        "The system should be used as an inspection-support tool."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    f"Developed by {NAME} | "
    f"{BRANCH} | "
    f"{COLLEGE}"
)

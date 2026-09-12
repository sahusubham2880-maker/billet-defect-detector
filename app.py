# ================================================================
# CONTINUOUS CASTING BILLET DEFECT DETECTOR
# ONE COMPLETE COLAB CODE
# ================================================================
#
# Developed by: Subham Sahu
# Registration No.: 2301105744
# Branch: Metallurgical & Materials Engineering
# College: Indira Gandhi Institute of Technology (IGIT), Sarang
#
# Core trained classes:
# 1. Scratch
# 2. Weld slag
# 3. Cutting opening
# 4. Water slag mark
# 5. Slag skin
# 6. Longitudinal crack
#
# IMPORTANT:
# This system reports REAL model performance from an unseen test set.
# Model confidence is NOT the same as accuracy/probability.
# It does not claim to detect untrained, microscopic or internal defects.
# ================================================================


# ================================================================
# 1. INSTALL LIBRARIES
# ================================================================

!pip install -q -U ultralytics streamlit opencv-python-headless pillow pandas numpy pyyaml scikit-learn matplotlib


# ================================================================
# 2. IMPORTS
# ================================================================

import os
import shutil
import zipfile
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from PIL import Image

from google.colab import files
from ultralytics import YOLO


# ================================================================
# 3. UPLOAD YOUR CASTING BILLET DATASET
# ================================================================

print("=" * 70)
print("UPLOAD YOUR CASTING BILLET DATASET ZIP")
print("=" * 70)

uploaded = files.upload()

zip_files = [
    x for x in uploaded.keys()
    if x.lower().endswith(".zip")
]

if not zip_files:
    raise RuntimeError(
        "Please upload your Casting Billet dataset as a ZIP file."
    )

ZIP_FILE = zip_files[0]

print("Uploaded:", ZIP_FILE)


# ================================================================
# 4. EXTRACT DATASET
# ================================================================

SOURCE_DIR = Path("/content/casting_source")

if SOURCE_DIR.exists():
    shutil.rmtree(SOURCE_DIR)

SOURCE_DIR.mkdir(parents=True)

with zipfile.ZipFile(ZIP_FILE, "r") as z:
    z.extractall(SOURCE_DIR)

print("Dataset extracted successfully.")


# ================================================================
# 5. FIND ORIGINAL DATASET FOLDERS
# ================================================================

def find_folder(root, folder_name):

    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() == folder_name.lower():
            return p

    return None


IMAGE_DIR = find_folder(
    SOURCE_DIR,
    "images"
)

LABEL_DIR = find_folder(
    SOURCE_DIR,
    "labels"
)

MASK_DIR = find_folder(
    SOURCE_DIR,
    "mask"
)

CLASS_FILE = None

for p in SOURCE_DIR.rglob("classes.txt"):
    CLASS_FILE = p
    break


print("\nDataset structure:")
print("Images :", IMAGE_DIR)
print("Labels :", LABEL_DIR)
print("Masks  :", MASK_DIR)
print("Classes:", CLASS_FILE)


if IMAGE_DIR is None:
    raise RuntimeError(
        "The ZIP does not contain an images folder."
    )

if LABEL_DIR is None:
    raise RuntimeError(
        "The ZIP does not contain a labels folder."
    )


# ================================================================
# 6. READ CLASS NAMES
# ================================================================

EXPECTED_CLASSES = [
    "scratch",
    "weld slag",
    "cutting opening",
    "water slag mark",
    "slag skin",
    "longitudinal crack"
]

classes = EXPECTED_CLASSES.copy()

if CLASS_FILE is not None:

    try:

        with open(
            CLASS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            loaded = [
                line.strip()
                for line in f
                if line.strip()
            ]

        if len(loaded) == len(EXPECTED_CLASSES):
            classes = loaded

    except Exception:
        classes = EXPECTED_CLASSES.copy()


print("\nClasses used by the model:")

for i, name in enumerate(classes):
    print(i, "=", name)


# ================================================================
# 7. FIND IMAGES
# ================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}

all_images = []

for p in IMAGE_DIR.rglob("*"):

    if (
        p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ):
        all_images.append(p)


print(
    "\nImages found:",
    len(all_images)
)

if not all_images:
    raise RuntimeError(
        "No images were found."
    )


# ================================================================
# 8. MATCH IMAGES WITH LABELS
# ================================================================

def get_label(image_path):

    stem = image_path.stem

    direct = LABEL_DIR / f"{stem}.txt"

    if direct.exists():
        return direct

    matches = list(
        LABEL_DIR.rglob(
            f"{stem}.txt"
        )
    )

    if matches:
        return matches[0]

    return None


pairs = []

missing = []

for image_path in all_images:

    label_path = get_label(
        image_path
    )

    if label_path is None:

        missing.append(
            image_path
        )

    else:

        pairs.append(
            (
                image_path,
                label_path
            )
        )


print(
    "Image-label pairs:",
    len(pairs)
)

print(
    "Missing labels:",
    len(missing)
)


if not pairs:
    raise RuntimeError(
        "No image-label pairs were found."
    )


# ================================================================
# 9. VALIDATE LABELS
# ================================================================

def check_label(label_path):

    try:

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = [
                x.strip()
                for x in f
                if x.strip()
            ]

        if not lines:
            return False

        for line in lines:

            parts = line.split()

            if len(parts) < 5:
                return False

            class_id = int(
                float(parts[0])
            )

            if (
                class_id < 0
                or class_id >= len(classes)
            ):
                return False

            values = [
                float(x)
                for x in parts[1:]
            ]

            # Detection label:
            # class x y width height
            if len(values) == 4:

                if not all(
                    0 <= x <= 1
                    for x in values
                ):
                    return False

            # Segmentation label:
            # class x1 y1 x2 y2 ...
            elif len(values) >= 6:

                if len(values) % 2 != 0:
                    return False

                if not all(
                    0 <= x <= 1
                    for x in values
                ):
                    return False

            else:
                return False

        return True

    except Exception:

        return False


valid_pairs = []

invalid_pairs = []

for image_path, label_path in pairs:

    if check_label(label_path):

        valid_pairs.append(
            (
                image_path,
                label_path
            )
        )

    else:

        invalid_pairs.append(
            (
                image_path,
                label_path
            )
        )


print(
    "\nValid labelled images:",
    len(valid_pairs)
)

print(
    "Invalid labelled images:",
    len(invalid_pairs)
)


if not valid_pairs:
    raise RuntimeError(
        "No valid labelled images remain."
    )


# ================================================================
# 10. DETERMINE DETECTION / SEGMENTATION
# ================================================================

segmentation_count = 0
detection_count = 0

for _, label_path in valid_pairs:

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        lines = [
            x.strip()
            for x in f
            if x.strip()
        ]

    for line in lines:

        number_of_values = len(
            line.split()
        ) - 1

        if number_of_values == 4:
            detection_count += 1

        elif number_of_values >= 6:
            segmentation_count += 1


USE_SEGMENTATION = (
    segmentation_count > 0
    and detection_count == 0
)


if USE_SEGMENTATION:

    MODEL_NAME = "yolo26n-seg.pt"

    print(
        "\nTraining mode: INSTANCE SEGMENTATION"
    )

else:

    MODEL_NAME = "yolo26n.pt"

    print(
        "\nTraining mode: OBJECT DETECTION"
    )

    print(
        "Existing detection annotations will be used."
    )


# ================================================================
# 11. CREATE CLEAN DATASET
# ================================================================

FINAL_DATASET = Path(
    "/content/billet_dataset"
)

if FINAL_DATASET.exists():
    shutil.rmtree(
        FINAL_DATASET
    )


for split in [
    "train",
    "val",
    "test"
]:

    (
        FINAL_DATASET
        / "images"
        / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        FINAL_DATASET
        / "labels"
        / split
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ================================================================
# 12. SPLIT DATA
# ================================================================

random.seed(42)

random.shuffle(
    valid_pairs
)

total = len(valid_pairs)

train_end = int(
    total * 0.70
)

val_end = int(
    total * 0.85
)

train_pairs = valid_pairs[
    :train_end
]

val_pairs = valid_pairs[
    train_end:val_end
]

test_pairs = valid_pairs[
    val_end:
]


print("\nDataset split:")
print("Train:", len(train_pairs))
print("Validation:", len(val_pairs))
print("Test:", len(test_pairs))


# ================================================================
# 13. COPY DATA
# ================================================================

def copy_split(
    pair_list,
    split
):

    for i, (
        image_path,
        label_path
    ) in enumerate(pair_list):

        image_name = (
            f"{i:06d}"
            + image_path.suffix.lower()
        )

        label_name = (
            f"{i:06d}.txt"
        )

        shutil.copy2(
            image_path,
            FINAL_DATASET
            / "images"
            / split
            / image_name
        )

        shutil.copy2(
            label_path,
            FINAL_DATASET
            / "labels"
            / split
            / label_name
        )


copy_split(
    train_pairs,
    "train"
)

copy_split(
    val_pairs,
    "val"
)

copy_split(
    test_pairs,
    "test"
)


# ================================================================
# 14. CREATE DATASET YAML
# ================================================================

DATA_YAML = (
    FINAL_DATASET
    / "dataset.yaml"
)

dataset_yaml = {

    "path": str(
        FINAL_DATASET
    ),

    "train": "images/train",

    "val": "images/val",

    "test": "images/test",

    "names": {
        i: name
        for i, name in enumerate(classes)
    }
}


with open(
    DATA_YAML,
    "w",
    encoding="utf-8"
) as f:

    yaml.safe_dump(
        dataset_yaml,
        f,
        sort_keys=False,
        allow_unicode=True
    )


# ================================================================
# 15. TRAIN MODEL
# ================================================================

print("\n")
print("=" * 70)
print("STARTING MODEL TRAINING")
print("=" * 70)

model = YOLO(
    MODEL_NAME
)


model.train(

    data=str(
        DATA_YAML
    ),

    epochs=80,

    imgsz=960,

    batch=8,

    patience=20,

    pretrained=True,

    device=0,

    workers=2,

    optimizer="auto",

    cos_lr=True,

    close_mosaic=10,

    degrees=0.0,

    translate=0.10,

    scale=0.40,

    fliplr=0.50,

    flipud=0.0,

    hsv_h=0.015,

    hsv_s=0.30,

    hsv_v=0.20,

    project="/content/billet_training",

    name="billet_defect_model",

    exist_ok=True,

    verbose=True
)


# ================================================================
# 16. LOAD BEST MODEL
# ================================================================

BEST_MODEL = (
    Path("/content/billet_training")
    / "billet_defect_model"
    / "weights"
    / "best.pt"
)


if not BEST_MODEL.exists():

    raise RuntimeError(
        "Training completed but best.pt was not found."
    )


best_model = YOLO(
    str(BEST_MODEL)
)


# ================================================================
# 17. FINAL UNSEEN TEST
# ================================================================

print("\n")
print("=" * 70)
print("FINAL UNSEEN TEST SET")
print("=" * 70)


metrics = best_model.val(

    data=str(
        DATA_YAML
    ),

    split="test",

    imgsz=960,

    batch=8,

    device=0,

    plots=True,

    verbose=True
)


# ================================================================
# 18. PERFORMANCE REPORT
# ================================================================

print("\n")
print("=" * 70)
print("REAL TEST PERFORMANCE")
print("=" * 70)


try:

    precision = float(
        metrics.box.mp
    )

    recall = float(
        metrics.box.mr
    )

    map50 = float(
        metrics.box.map50
    )

    map5095 = float(
        metrics.box.map
    )

    if (
        precision + recall
        > 0
    ):

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    else:

        f1 = 0


    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"mAP@50    : {map50:.4f}"
    )

    print(
        f"mAP@50-95 : {map5095:.4f}"
    )


except Exception as error:

    print(
        "Could not extract metrics:",
        error
    )


# ================================================================
# 19. PER-CLASS PERFORMANCE
# ================================================================

print("\n")
print("=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)


try:

    for class_id, class_name in (
        best_model.names.items()
    ):

        try:

            p = float(
                metrics.box.p[class_id]
            )

            r = float(
                metrics.box.r[class_id]
            )

            ap50 = float(
                metrics.box.ap50[class_id]
            )

            ap = float(
                metrics.box.ap[class_id]
            )

            if p + r > 0:

                class_f1 = (
                    2 * p * r
                    / (p + r)
                )

            else:

                class_f1 = 0


            print(
                f"\n{class_name}"
            )

            print(
                f"  Precision : {p:.4f}"
            )

            print(
                f"  Recall    : {r:.4f}"
            )

            print(
                f"  F1        : {class_f1:.4f}"
            )

            print(
                f"  AP50      : {ap50:.4f}"
            )

            print(
                f"  AP50-95   : {ap:.4f}"
            )

        except Exception:
            pass


except Exception as error:

    print(
        "Per-class metrics unavailable:",
        error
    )


# ================================================================
# 20. SAVE FINAL MODEL
# ================================================================

FINAL_MODEL = Path(
    "/content/best_billet_defect_model.pt"
)

shutil.copy2(
    BEST_MODEL,
    FINAL_MODEL
)


# ================================================================
# 21. CREATE STREAMLIT APP
# ================================================================

APP_CODE = r'''
import streamlit as st

from ultralytics import YOLO

from PIL import Image

import numpy as np
import pandas as pd

import cv2
import io

from datetime import datetime


# ==========================================================
# PAGE
# ==========================================================

st.set_page_config(
    page_title="Billet Defect Inspector",
    page_icon="🔍",
    layout="wide"
)


# ==========================================================
# MODEL
# ==========================================================

MODEL_PATH = (
    "best_billet_defect_model.pt"
)


CLASS_NAMES = [
    "scratch",
    "weld slag",
    "cutting opening",
    "water slag mark",
    "slag skin",
    "longitudinal crack"
]


@st.cache_resource
def load_model():

    return YOLO(
        MODEL_PATH
    )


model = load_model()


# ==========================================================
# SIDEBAR — PERSONAL DETAILS
# ==========================================================

st.sidebar.title(
    "👨‍💻 Developed by"
)

st.sidebar.markdown(
    """
**Subham Sahu**

**Registration No.:** 2301105744

**Branch:** Metallurgical & Materials Engineering

**College:** Indira Gandhi Institute of Technology (IGIT), Sarang
"""
)


st.sidebar.divider()


# ==========================================================
# SIDEBAR — SETTINGS
# ==========================================================

st.sidebar.subheader(
    "⚙ Inspection Settings"
)


CONFIDENCE = st.sidebar.slider(
    "Minimum model confidence",
    0.05,
    0.95,
    0.25,
    0.05
)


IOU = st.sidebar.slider(
    "IoU threshold",
    0.10,
    0.90,
    0.50,
    0.05
)


TILE_SIZE = st.sidebar.selectbox(
    "High-resolution tile size",
    [640, 768, 960, 1280],
    index=2
)


OVERLAP = st.sidebar.slider(
    "Tile overlap",
    0.10,
    0.50,
    0.20,
    0.05
)


USE_TILED = st.sidebar.checkbox(
    "High-resolution inspection",
    True
)


# ==========================================================
# MAIN PAGE
# ==========================================================

st.title(
    "🔍 Continuous Casting Billet Defect Inspector"
)


st.write(
    "AI-assisted surface inspection of continuous-casting "
    "billets using the trained defect-detection model."
)


st.info(
    "Model confidence is a model score, not a certified "
    "probability or accuracy."
)


# ==========================================================
# IMAGE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload Billet Surface Image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tif",
        "tiff",
        "webp"
    ]
)


# ==========================================================
# TILED INFERENCE
# ==========================================================

def tiled_inference(
    image,
    tile_size,
    overlap,
    confidence,
    iou
):

    image_np = np.array(
        image.convert("RGB")
    )

    height, width = (
        image_np.shape[:2]
    )

    bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    stride = max(
        1,
        int(
            tile_size
            * (1 - overlap)
        )
    )


    x_positions = list(
        range(
            0,
            max(
                1,
                width - tile_size + 1
            ),
            stride
        )
    )


    y_positions = list(
        range(
            0,
            max(
                1,
                height - tile_size + 1
            ),
            stride
        )
    )


    if (
        not x_positions
        or x_positions[-1]
        + tile_size < width
    ):

        x_positions.append(
            max(
                0,
                width - tile_size
            )
        )


    if (
        not y_positions
        or y_positions[-1]
        + tile_size < height
    ):

        y_positions.append(
            max(
                0,
                height - tile_size
            )
        )


    boxes = []
    scores = []
    class_ids = []


    for y in y_positions:

        for x in x_positions:

            tile = bgr[
                y:min(
                    y + tile_size,
                    height
                ),
                x:min(
                    x + tile_size,
                    width
                )
            ]


            if tile.size == 0:
                continue


            results = model.predict(
                tile,
                imgsz=tile_size,
                conf=confidence,
                iou=iou,
                verbose=False
            )


            result = results[0]


            if result.boxes is None:
                continue


            for box in result.boxes:

                coords = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )


                score = float(
                    box.conf[0]
                    .cpu()
                    .numpy()
                )


                class_id = int(
                    box.cls[0]
                    .cpu()
                    .numpy()
                )


                x1, y1, x2, y2 = coords


                boxes.append([
                    float(x1 + x),
                    float(y1 + y),
                    float(x2 + x),
                    float(y2 + y)
                ])


                scores.append(
                    score
                )


                class_ids.append(
                    class_id
                )


    # ======================================================
    # MERGE OVERLAPPING TILE DETECTIONS
    # ======================================================

    final_indices = []


    if boxes:

        nms_boxes = []


        for (
            x1,
            y1,
            x2,
            y2
        ) in boxes:

            nms_boxes.append([
                int(x1),
                int(y1),
                int(x2 - x1),
                int(y2 - y1)
            ])


        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            scores,
            confidence,
            iou
        )


        if len(indices) > 0:

            final_indices = (
                np.array(
                    indices
                )
                .reshape(-1)
                .tolist()
            )


    detections = []


    for index in final_indices:

        detections.append({
            "class_id":
                class_ids[index],

            "confidence":
                scores[index],

            "box":
                boxes[index]
        })


    return (
        image_np,
        detections
    )


# ==========================================================
# INSPECTION
# ==========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    image_np = np.array(
        image
    )


    height, width = (
        image_np.shape[:2]
    )


    st.subheader(
        "📷 Billet Image"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )


    # ======================================================
    # RUN INSPECTION
    # ======================================================

    if st.button(
        "🔎 Run AI Inspection",
        type="primary"
    ):


        with st.spinner(
            "Analyzing billet surface..."
        ):


            # ------------------------------------------------
            # IMAGE QUALITY
            # ------------------------------------------------

            gray = cv2.cvtColor(
                image_np,
                cv2.COLOR_RGB2GRAY
            )


            sharpness = cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            if USE_TILED:

                inspected_image, detections = (
                    tiled_inference(
                        image,
                        TILE_SIZE,
                        OVERLAP,
                        CONFIDENCE,
                        IOU
                    )
                )

            else:

                result = model.predict(
                    image,
                    imgsz=960,
                    conf=CONFIDENCE,
                    iou=IOU,
                    verbose=False
                )[0]


                inspected_image = (
                    image_np.copy()
                )


                detections = []


                if result.boxes is not None:

                    for box in result.boxes:

                        coords = (
                            box.xyxy[0]
                            .cpu()
                            .numpy()
                        )


                        score = float(
                            box.conf[0]
                            .cpu()
                            .numpy()
                        )


                        class_id = int(
                            box.cls[0]
                            .cpu()
                            .numpy()
                        )


                        detections.append({
                            "class_id":
                                class_id,

                            "confidence":
                                score,

                            "box":
                                coords.tolist()
                        })


            # ------------------------------------------------
            # DRAW DETECTIONS
            # ------------------------------------------------

            annotated = (
                inspected_image.copy()
            )


            rows = []


            for number, detection in enumerate(
                detections,
                start=1
            ):


                class_id = (
                    detection["class_id"]
                )


                confidence = (
                    detection["confidence"]
                )


                x1, y1, x2, y2 = [
                    int(v)
                    for v in detection["box"]
                ]


                x1 = max(
                    0,
                    min(
                        width - 1,
                        x1
                    )
                )


                x2 = max(
                    0,
                    min(
                        width - 1,
                        x2
                    )
                )


                y1 = max(
                    0,
                    min(
                        height - 1,
                        y1
                    )
                )


                y2 = max(
                    0,
                    min(
                        height - 1,
                        y2
                    )
                )


                box_width = max(
                    0,
                    x2 - x1
                )


                box_height = max(
                    0,
                    y2 - y1
                )


                area = (
                    box_width
                    * box_height
                )


                if (
                    0 <= class_id
                    < len(CLASS_NAMES)
                ):

                    defect_name = (
                        CLASS_NAMES[
                            class_id
                        ]
                    )

                else:

                    defect_name = (
                        f"Class {class_id}"
                    )


                # Draw rectangle
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    3
                )


                # Label
                label = (
                    f"{defect_name} "
                    f"| {confidence:.3f}"
                )


                cv2.putText(
                    annotated,
                    label,
                    (
                        x1,
                        max(
                            25,
                            y1 - 8
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA
                )


                rows.append({

                    "Defect No.":
                        number,

                    "Defect":
                        defect_name,

                    "Model Confidence":
                        round(
                            confidence,
                            4
                        ),

                    "X1":
                        x1,

                    "Y1":
                        y1,

                    "X2":
                        x2,

                    "Y2":
                        y2,

                    "Width (px)":
                        box_width,

                    "Height (px)":
                        box_height,

                    "Area (px²)":
                        area
                })


            # ==================================================
            # RESULT IMAGE
            # ==================================================

            with col2:

                st.image(
                    annotated,
                    caption="AI Inspection Result",
                    use_container_width=True
                )


            # ==================================================
            # SUMMARY
            # ==================================================

            st.divider()


            st.subheader(
                "📊 Inspection Summary"
            )


            total_defects = len(
                rows
            )


            c1, c2, c3 = st.columns(3)


            c1.metric(
                "Detected Defects",
                total_defects
            )


            c2.metric(
                "Image Width",
                f"{width}px"
            )


            c3.metric(
                "Image Height",
                f"{height}px"
            )


            if total_defects == 0:

                st.success(
                    "No trained defect class was detected "
                    "above the selected confidence threshold."
                )


            else:

                st.subheader(
                    "🔍 Defect Details"
                )


                df = pd.DataFrame(
                    rows
                )


                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )


                # ==================================================
                # DEFECT COUNTS
                # ==================================================

                st.subheader(
                    "📌 Defect Count"
                )


                counts = (
                    df["Defect"]
                    .value_counts()
                    .rename_axis(
                        "Defect"
                    )
                    .reset_index(
                        name="Count"
                    )
                )


                st.dataframe(
                    counts,
                    use_container_width=True,
                    hide_index=True
                )


                # ==================================================
                # HIGHEST SCORE
                # ==================================================

                highest = (
                    df.sort_values(
                        "Model Confidence",
                        ascending=False
                    )
                    .iloc[0]
                )


                st.info(
                    "Highest model-confidence detection: "
                    f"{highest['Defect']} "
                    f"({highest['Model Confidence']:.3f})"
                )


                # ==================================================
                # CSV REPORT
                # ==================================================

                csv_data = (
                    df.to_csv(
                        index=False
                    )
                )


                st.download_button(
                    "⬇ Download CSV Inspection Report",
                    data=csv_data,
                    file_name=(
                        "billet_inspection_report.csv"
                    ),
                    mime="text/csv"
                )


            # ==================================================
            # ANNOTATED IMAGE DOWNLOAD
            # ==================================================

            output = Image.fromarray(
                annotated
            )


            buffer = io.BytesIO()


            output.save(
                buffer,
                format="PNG"
            )


            st.download_button(
                "⬇ Download Annotated Image",
                data=buffer.getvalue(),
                file_name=(
                    "billet_defect_inspection.png"
                ),
                mime="image/png"
            )


            # ==================================================
            # IMAGE QUALITY
            # ==================================================

            st.divider()


            st.subheader(
                "🔬 Image Quality"
            )


            st.write(
                f"Resolution: "
                f"{width} × {height} pixels"
            )


            st.write(
                f"Sharpness indicator: "
                f"{sharpness:.2f}"
            )


            st.caption(
                "Sharpness is an image-quality indicator only "
                "and is not an industrial acceptance limit."
            )


# ==========================================================
# ENGINEERING NOTE
# ==========================================================

st.divider()


st.subheader(
    "⚠️ Engineering Note"
)


st.write(
    "This AI system detects the defect classes represented "
    "in its training dataset. It should not be interpreted "
    "as detecting every possible billet defect."
)


st.write(
    "Very small or microscopic defects require sufficient "
    "image resolution, suitable lighting and representative "
    "annotated training data."
)


st.write(
    "Internal or subsurface defects cannot be reliably "
    "determined from an ordinary surface photograph."
)


st.caption(
    "AI-assisted inspection prototype"
)
'''


# ================================================================
# 22. SAVE APP
# ================================================================

APP_PATH = Path(
    "/content/app.py"
)

with open(
    APP_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        APP_CODE
    )


# ================================================================
# 23. CREATE REQUIREMENTS
# ================================================================

REQUIREMENTS = """streamlit
ultralytics
opencv-python-headless
pillow
numpy
pandas
"""

REQUIREMENTS_PATH = Path(
    "/content/requirements.txt"
)

with open(
    REQUIREMENTS_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        REQUIREMENTS
    )


# ================================================================
# 24. CREATE FINAL PROJECT
# ================================================================

PROJECT_DIR = Path(
    "/content/billet_defect_detector"
)

if PROJECT_DIR.exists():

    shutil.rmtree(
        PROJECT_DIR
    )

PROJECT_DIR.mkdir()


shutil.copy2(
    APP_PATH,
    PROJECT_DIR / "app.py"
)


shutil.copy2(
    FINAL_MODEL,
    PROJECT_DIR
    / "best_billet_defect_model.pt"
)


shutil.copy2(
    REQUIREMENTS_PATH,
    PROJECT_DIR
    / "requirements.txt"
)


shutil.copy2(
    DATA_YAML,
    PROJECT_DIR
    / "dataset.yaml"
)


# ================================================================
# 25. CREATE README
# ================================================================

README = """
# Continuous Casting Billet Defect Detector

AI-assisted billet surface inspection.

## Trained classes

1. Scratch
2. Weld slag
3. Cutting opening
4. Water slag mark
5. Slag skin
6. Longitudinal crack

## Features

- YOLO-based defect detection
- High-resolution tiled inference
- Defect type
- Model confidence score
- Defect location
- Bounding-box dimensions
- Defect count
- CSV inspection report
- Annotated image
- Image-quality indicator
- Unseen test-set evaluation

## Important engineering limitation

The model can detect only classes represented in its training data.

Model confidence is not the same as accuracy or a calibrated probability.

Internal/subsurface defects should not be claimed as detectable
from an ordinary surface image.

Production deployment requires validation using representative
plant images and applicable inspection specifications.
"""

with open(
    PROJECT_DIR / "README.md",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        README
    )


# ================================================================
# 26. CREATE COMPLETE ZIP
# ================================================================

OUTPUT_BASE = (
    "/content/billet_defect_detector_complete"
)

OUTPUT_ZIP = (
    OUTPUT_BASE + ".zip"
)

if os.path.exists(
    OUTPUT_ZIP
):

    os.remove(
        OUTPUT_ZIP
    )


shutil.make_archive(
    OUTPUT_BASE,
    "zip",
    PROJECT_DIR
)


# ================================================================
# 27. FINAL INFORMATION
# ================================================================

print("\n")
print("=" * 70)
print("      COMPLETE BILLET DEFECT DETECTOR READY")
print("=" * 70)

print("\nDataset:")
print(
    "Valid images:",
    len(valid_pairs)
)

print(
    "Training:",
    len(train_pairs)
)

print(
    "Validation:",
    len(val_pairs)
)

print(
    "Unseen test:",
    len(test_pairs)
)

print("\nModel:")
print(
    "Instance Segmentation"
    if USE_SEGMENTATION
    else "Object Detection"
)

print("\nBest model:")
print(
    FINAL_MODEL
)

print("\nStreamlit app:")
print(
    APP_PATH
)

print("\nComplete project:")
print(
    OUTPUT_ZIP
)

print("\n")
print("=" * 70)
print("DOWNLOADING COMPLETE PROJECT")
print("=" * 70)


# ================================================================
# 28. DOWNLOAD
# ================================================================

files.download(
    OUTPUT_ZIP
)

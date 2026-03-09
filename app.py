import streamlit as st
import numpy as np
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity
from rembg import remove
from io import BytesIO

st.set_page_config(page_title="E-Commerce Visual Search", layout="wide")
st.write("#NOTE: it uses cosine similarity to find similar products, so it may not be perfect. For best results, use clear product images with minimal background.")
DATASET_PATH = "dataset"


# ----------------------------
# Feature Extraction
# ----------------------------
def extract_features(image):
    image = image.resize((100, 100))
    arr = np.array(image)

    histogram = []
    for i in range(3):
        hist, _ = np.histogram(arr[:, :, i], bins=32, range=(0, 256))
        histogram.extend(hist)

    histogram = np.array(histogram)

    norm = np.linalg.norm(histogram)
    if norm != 0:
        histogram = histogram / norm

    return histogram


# ----------------------------
# Load Dataset (Safe Version)
# ----------------------------
def load_dataset():
    features = []
    image_paths = []

    if not os.path.exists(DATASET_PATH):
        return np.array([]), []

    for file in os.listdir(DATASET_PATH):

        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(DATASET_PATH, file)

        try:
            img = Image.open(path).convert("RGB")
            feat = extract_features(img)
            features.append(feat)
            image_paths.append(path)
        except:
            continue

    return np.array(features), image_paths


# ----------------------------
# Cache Dataset (Prevents Reload Freeze)
# ----------------------------
@st.cache_resource
def cached_dataset():
    return load_dataset()


dataset_features, dataset_paths = cached_dataset()


# ----------------------------
# UI
# ----------------------------
st.title("🛍️ E-Commerce Visual Search System")
st.write("Upload a product image to find visually similar products.")

uploaded_file = st.file_uploader(
    "Upload Product Image", 
    type=["jpg", "png", "jpeg"]
)


# ----------------------------
# Query Handling
# ----------------------------
if uploaded_file:

    if len(dataset_paths) == 0:
        st.error("Dataset folder is empty or not found.")
    else:
        # Load uploaded image
        input_image = Image.open(uploaded_file).convert("RGB")

        # Background removal
        output = remove(input_image)

        # Handle both rembg return types
        if isinstance(output, Image.Image):
            query_img = output.convert("RGB")
        else:
            query_img = Image.open(BytesIO(output)).convert("RGB")

        st.image(query_img, caption="Background Removed Image", width=300)

        # Extract features
        query_features = extract_features(query_img)

        # Compute similarity
        similarities = cosine_similarity(
            [query_features], dataset_features
        )[0]

        # Top 5 matches
        top_indices = similarities.argsort()[-5:][::-1]

        st.subheader("Top Matching Products")

        cols = st.columns(5)
        for i, idx in enumerate(top_indices):
            with cols[i]:
                st.image(
                    dataset_paths[idx],
                    caption=f"Score: {similarities[idx]:.2f}"
                )
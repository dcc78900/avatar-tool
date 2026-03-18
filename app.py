import streamlit as st
import numpy as np
import cv2
from PIL import Image
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

st.title("头像去重工具（网页版）")

@st.cache_resource
def load_model():
    app = FaceAnalysis()
    app.prepare(ctx_id=0)
    return app

app = load_model()

def get_embedding(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    faces = app.get(img)
    if len(faces) > 0:
        return faces[0].embedding
    return None

uploaded_files = st.file_uploader(
    "上传头像（可多选）",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    embeddings = []
    names = []
    images = []

    for file in uploaded_files:
        image = Image.open(file)
        emb = get_embedding(image)
        if emb is not None:
            embeddings.append(emb)
            names.append(file.name)
            images.append(image)

    st.success(f"检测到 {len(embeddings)} 张人脸")

    threshold = 0.8
    used = set()

    for i in range(len(embeddings)):
        if i in used:
            continue

        group = [i]
        used.add(i)

        for j in range(i + 1, len(embeddings)):
            if j in used:
                continue

            sim = cosine_similarity(
                [embeddings[i]],
                [embeddings[j]]
            )[0][0]

            if sim > threshold:
                group.append(j)
                used.add(j)

        if len(group) > 1:
            st.subheader("发现重复头像 👇")
            cols = st.columns(len(group))
            for idx, col in zip(group, cols):
                col.image(images[idx], caption=names[idx])

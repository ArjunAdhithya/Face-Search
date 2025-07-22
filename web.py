import streamlit as st
import numpy as np 
from deepface import DeepFace
import chromadb
import os
import PIL.Image as Image


client = chromadb.PersistentClient(path="db/chroma_db")
collection = client.get_collection(name="face_embeddings")

st.title("Face Search")

query_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
threshold = st.slider("Distance Threshold", 0.0, 1.0, 0.3)

if query_image :
    temp_path = "temp_query.jpg"
    with open(temp_path, "wb") as f:
        f.write(query_image.read())

    raw = DeepFace.represent(temp_path, model_name="Facenet")
    emb = np.array(raw[0]['embedding'])
    emb = emb / np.linalg.norm(emb)

    results = collection.query(query_embeddings=[emb.tolist()], n_results=5)

    st.image(temp_path, caption="Query Image", width=200)
    st.subheader("Matched Faces")

    found = False
    matched_images = []
    captions = []
    
    for img_id, dist in zip(results["ids"][0], results["distances"][0]):
        if dist < threshold and os.path.exists(img_id):
            img = Image.open(img_id)
            matched_images.append(img)
            name = os.path.splitext(os.path.basename(img_id))[0]
            captions.append(f"{name}\nDistance: {dist:.4f}")
    
    if matched_images:
        st.image(matched_images, caption=captions, width=200)
    else:
        st.warning("No matching faces found below the threshold")

st.subheader("Database Images")

all_results = collection.get()
if all_results and "ids" in all_results:
    cols = st.columns(5)
    for idx, img_path in enumerate(all_results["ids"]):
        with cols[idx % 5]:
            if os.path.exists(img_path):
                try :
                    img = Image.open(img_path)
                    st.image(img, caption=os.path.basename(img_path), width=150)
                except Exception as e:
                    st.write(f"Could not open {img_path}: {e}")

import os 
import datapath
from deepface import DeepFace

def embed_images():
    images = datapath.image_paths
    embeddings =[]
    
    for image_path in images:
        try:
            full_embedding = DeepFace.represent(image_path, model_name = 'Facenet')
            if isinstance(full_embedding, list):
                embedding = full_embedding[0]['embedding']
                embeddings.append((image_path,embedding))
                print(F"Embedding : {image_path} done")
        except Exception as e:
            print(f"Error in embedding images: {e}")

    return embeddings
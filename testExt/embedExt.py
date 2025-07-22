import os 
import numpy as np
import datapathExt
from deepface import DeepFace

def embed_images():
    images = datapathExt.image_paths
    embeddings =[]
    
    for image_path in images:
        try:
            full_embedding = DeepFace.represent(image_path, model_name = 'Facenet')
            if isinstance(full_embedding, list):
                embedding = np.array(full_embedding[0]['embedding'])
                embedding = embedding/np.linalg.norm(embedding)  
                embeddings.append((image_path,embedding.tolist()))
                print(F"Embedding : {image_path} done")
        except Exception as e:
            print(f"Error in embedding images: {e}")

    return embeddings

def multiface_embed():
    images = datapathExt.image_paths
    embeddings = []

    for image_path in images:
        try:
            full_embeddings = DeepFace.represent(image_path, model_name='Facenet')
            if isinstance(full_embeddings, list):
                for face_data in full_embeddings:
                    embeddings.append((image_path, face_data['embedding']))
                    print(f"Embedded face from {image_path}")
        except Exception as e:
            print(f"Error in embedding {image_path}: {e}")

    return embeddings
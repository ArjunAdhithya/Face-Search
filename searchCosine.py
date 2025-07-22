import chromadb
import os 
from deepface import DeepFace 
import numpy as np

client = chromadb.PersistentClient(path="db/chroma_db")
collection = client.get_collection(name = "face_embeddings")

query_image = input("Enter img with path: ").strip()
if not os.path.exists(query_image):
    print("Error not img found")

full_embedding = DeepFace.represent(query_image, model_name = 'Facenet')
query_embedding = full_embedding[0]['embedding']

results = collection.query(query_embeddings=[query_embedding], n_results=5)
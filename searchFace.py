import os
import chromadb
from deepface import DeepFace

client = chromadb.PersistentClient(path="db/chroma_db")
collection = client.get_collection(name="face_embeddings")

query_image_path = r"./targetIMGs/nelson2.jpg"
print(query_image_path)

raw = DeepFace.represent(query_image_path, model_name='Facenet')
query_embedding = raw[0]['embedding']

results = collection.query(query_embeddings=[query_embedding], n_results=5)

'''
for img_path, distance in zip(results['ids'][0], results['distances'][0]):
    print(f"Image Path: {img_path}, Distance: {distance}")
'''

treshold = 0.3
matches = [(img_path,distance) for img_path,distance in zip(results['ids'][0], results['distances'][0]) 
           if distance < treshold]
if matches:
    print("Person found")
    for img_path, distance in zip(results['ids'][0], results['distances'][0]):
        print(f"Image Path: {img_path}, Distance: {distance}")
else:
    print("Person not found")


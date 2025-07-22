import os
import datapath
import embed
import chromadb 
client = chromadb.PersistentClient(path="db/chroma_db")
collection = client.get_or_create_collection(name="face_embeddings")

embeddings = embed.embed_images()  
print(f"Total embeddings created: {len(embeddings)}")

for image_path, embedding in embeddings:
    collection.add(embeddings=[embedding],ids=[image_path])


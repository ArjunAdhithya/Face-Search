import chromadb

client = chromadb.PersistentClient(path="db/chroma_db")
collection = client.get_collection(name="face_embeddings")

#print(collection.count())


results = collection.get(include=['embeddings'])

for img_id,embedding in zip(results['ids'], results['embeddings']):
    print(f"Image ID: {img_id}, Embedding: {embedding[:5]}")  

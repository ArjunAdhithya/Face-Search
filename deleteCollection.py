import chromadb
client = chromadb.PersistentClient(path="db/chroma_db")
# client.delete_collection(name="face_embeddings")
# print("Collection 'face_embeddings' deleted successfully.")
collection = client.get_collection(name="face_embeddings")
print(collection.count())
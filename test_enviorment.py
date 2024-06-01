from sentence_transformers import SentenceTransformer
model = SentenceTransformer("./all-MiniLM-L6-v2")
sentences = [
    "The weather is so nice!",
     "It's so sunny outside.",
    "He's driving to the movie theater.",
    "She's going to the cinema.",
 ]
embeddings = model.encode(sentences, normalize_embeddings=True)
print(embeddings.shape)
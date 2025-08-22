import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding import BGEM3FlagModel

# Caricare il modello
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Frasi da codificare
frasi = [
    "Come creare contenuti youtube",
    "Quanti cavalli ha la Ferrari",
    "Chi è Michael Shumacher",
    "Video Youtube",
    "Iscriviti al canale di Ignis Forge",
    "Ultimo modello di OpenAi",
    "Il meccanico mi ha riparato l'automobile",
    "Ultimo modello della BMW"
]

# Variabile query da embeddare
query = "Quanto costa Tesla model S?"

# Ottenere gli embeddings delle frasi
embeddings = model.encode(frasi)['dense_vecs']

# Ottenere l'embedding della query
query_embedding = model.encode([query])['dense_vecs']

# Aggiungere l'embedding della query agli embeddings esistenti
embeddings = np.vstack([embeddings, query_embedding])

np.save("embeddings.npy", embeddings)
load_embeddings = np.load("embeddings.npy")

similarities = cosine_similarity(query_embedding, embeddings[:-1])[0]  # Escludiamo la query stessa

# Ordinare le frasi in base alla similarità in ordine decrescente
sorted_indices = np.argsort(similarities)[::-1]
sorted_similarities = similarities[sorted_indices]
sorted_frasi = [frasi[i] for i in sorted_indices]

# Stampare i risultati
print("Risultati ordinati per similarità:")
for frase, similarity in zip(sorted_frasi, sorted_similarities):
    print(f"{frase}: {similarity:.4f}")

# Ridurre la dimensionalità con t-SNE
tsne = TSNE(n_components=2, perplexity=2, random_state=42)
embeddings_2d = tsne.fit_transform(load_embeddings)

# Plottare i punti
plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker='o', color='blue')
plt.scatter(embeddings_2d[-1, 0], embeddings_2d[-1, 1], marker='o', color='red')  # punto della query


# Annotare i punti con le frasi corrispondenti
for i, frase in enumerate(frasi):
    plt.annotate(frase, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=9, alpha=0.75)
    

# Annotare la query
plt.annotate(query, (embeddings_2d[-1, 0], embeddings_2d[-1, 1]), fontsize=9, alpha=0.75, color='red')

# Aggiungere la legenda
plt.legend()

plt.title("Visualizzazione t-SNE degli embeddings")
plt.xlabel("Dimensione 1")
plt.ylabel("Dimensione 2")
plt.grid(True)
plt.show()

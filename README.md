# Retrieval Augmented Generation


![rag.png](/rag.png)


## Funzionamento


![vettorializzazione.png](/vettorializzazione.png)

![what-are-embeddings.svg](/what-are-embeddings.svg)

Il testo vettorializzato viene disposto in uno spazio vettoriale.
I modelli utilizzano uno spazio vettoriale di 1024 dimensioni

Ogni puntino nello spazio vettoriale è disposto in modo "semantico"

![rag-plot.png](/rag-plot.png)

![2025-02-19_17-55.png](/ollama_rag/2025-02-19_17-55.png)

## Embedding Python

[Modello di embedding](https://huggingface.co/BAAI/bge-m3)

Plot delle seguenti frasi embeddate : 
- "Come creare contenuti youtube",
- "Quanti cavalli ha la Ferrari",
- "Chi è Michael Shumacher",
- "Video Youtube",
- "Iscriviti al canale di Ignis Forge",
- "Ultimo modello di OpenAi",
- "Il meccanico mi ha riparato l'automobile",
- "Ultimo modello della BMW"

![2025-02-19_18-42.png](/2025-02-19_18-42.png)

La query dell'utente viene posizionata all'interno dell grafico 

![2025-02-19_22-25.png](/2025-02-19_22-25.png)

## Cosine similarity e distanza

Troviamo il punto più pertinente

![cosine-similarity.webp](/cosine-similarity.webp)

![download.png](/cosine-similarity-sample.png)

![2025-02-19_22-43.png](/ollama_rag/2025-02-19_22-43.png)

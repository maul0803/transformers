from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import time

def save_time(func_name, elapsed_time):
    with open("results_times.txt", "a") as f:
        f.write(f"{func_name} executed in {elapsed_time:.4f} seconds\n")

def get_paragraphs(size="100k"):
    start_time = time.time()
    file_path = f"paragraphs_{size}.txt"
    paragraphs = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            paragraphs.append(line.strip())
    
    elapsed_time = time.time() - start_time
    save_time("get_paragraphs", elapsed_time)
    return paragraphs

def get_embeddings(model, paragraphs, embeddings_file):
    start_time = time.time()
    if os.path.exists(embeddings_file):
        embeddings = np.load(embeddings_file)
    else:
        embeddings = model.encode(paragraphs, show_progress_bar=True, normalize_embeddings=True).astype('float32')
        np.save(embeddings_file, embeddings)
    
    elapsed_time = time.time() - start_time
    save_time("get_embeddings", elapsed_time)
    return embeddings

def get_centroids(k, embeddings, d, centroids_file):
    start_time = time.time()
    if os.path.exists(centroids_file):
        centroids = np.load(centroids_file)
    else:
        res = faiss.StandardGpuResources()
        embeddings = embeddings.astype(np.float32)

        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 0

        index = faiss.GpuIndexFlatL2(res, d, flat_config)

        clustering = faiss.Clustering(d, k)
        clustering.train(embeddings, index)

        centroids = faiss.vector_to_array(clustering.centroids).reshape(k, d)
        np.save(centroids_file, centroids)
    
    elapsed_time = time.time() - start_time
    save_time("get_centroids", elapsed_time)
    return centroids

def get_faiss_index_IVFFlat(embeddings, centroids):
    start_time = time.time()
    faiss_index_base = faiss.IndexFlatIP(embeddings.shape[1])
    nlist = int(centroids.shape[0] / 40)
    faiss_index = faiss.IndexIVFFlat(faiss_index_base, embeddings.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)

    faiss_index.train(centroids)
    faiss_index.add(embeddings)
    
    elapsed_time = time.time() - start_time
    save_time("get_faiss_index_IVFFlat", elapsed_time)
    return faiss_index

def get_faiss_index_IVFPQ(embeddings, centroids):
    start_time = time.time()
    
    num_points = embeddings.shape[0]
    dim = embeddings.shape[1]

    # Nombre de clusters (nlist) respectant la règle des 40 points par cluster
    nlist = max(1, int(num_points / 40))

    # PQ parameters
    M = 32  # Number of subquantizers (should divide `dim`)
    n_bits = 8

    if dim % M != 0:
        raise ValueError(f"Embedding dimension {dim} must be divisible by M={M} for PQ.")

    print("Training IVF-PQ with:")
    print(f"  embeddings: {embeddings.shape}")
    print(f"  nlist: {nlist}")
    print(f"  M: {M}")
    print(f"  n_bits: {n_bits}")

    quantizer = faiss.IndexFlatL2(dim)
    index_ivf = faiss.IndexIVFPQ(quantizer, dim, nlist, M, n_bits)

    # Entraînement sur les embeddings (et pas les centroids !)
    index_ivf.train(embeddings)
    index_ivf.add(embeddings)

    elapsed_time = time.time() - start_time
    save_time("get_faiss_index_IVFPQ", elapsed_time)
    
    return index_ivf

def find_similar_paragraphs(faiss_index, embeddings, threshold=0.75):
    start_time = time.time()
    limits, distances, indices = faiss_index.range_search(x=embeddings, thresh=threshold)
    
    elapsed_time = time.time() - start_time
    save_time("find_similar_paragraphs", elapsed_time)
    return limits, distances, indices

def main(size="100k", model_name="all-mpnet-base-v2"):
    paragraphs = get_paragraphs(size)
    model = SentenceTransformer(model_name)
    embeddings = get_embeddings(model, paragraphs, f"embeddings_{size}.npy")

    k = int(embeddings.shape[0] / 40)
    d = embeddings.shape[1]

    centroids = get_centroids(k, embeddings, d, f"centroids_{size}.npy")

    faiss_index = get_faiss_index_IVFFlat(embeddings, centroids)
    #faiss_index = get_faiss_index_IVFPQ(embeddings, centroids)
    limits, distances, indices = find_similar_paragraphs(faiss_index, embeddings, threshold=0.75)

    for i in range(len(embeddings)-1, len(embeddings)):
        print(f"Vector {i}:")
        start = limits[i]
        end = limits[i+1]
        buffer = 0 
        for j in range(start, end):
            buffer += 1
            if indices[j] != i:
                print(f"  Neighbor {indices[j]} with a distance of {distances[j]:.4f}")
            if buffer >= 10:
                break

if __name__ == "__main__":
    main()

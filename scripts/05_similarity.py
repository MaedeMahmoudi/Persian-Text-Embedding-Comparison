import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

output_folder = "../outputs"

with open(os.path.join(output_folder, 'tfidf_matrix.pkl'), 'rb') as f:
    tfidf_matrix = pickle.load(f)

with open(os.path.join(output_folder, 'reduced_matrix.pkl'), 'rb') as f:
    reduced_matrix = pickle.load(f)

with open(os.path.join(output_folder, 'embedding_matrix.pkl'), 'rb') as f:
    embedding_matrix = pickle.load(f)

tfidf_sim = cosine_similarity(tfidf_matrix)
svd_sim = cosine_similarity(reduced_matrix)
embedding_sim = cosine_similarity(embedding_matrix)

with open(os.path.join(output_folder, 'similarity_results.txt'), 'w', encoding='utf-8') as f:
    f.write("TF-IDF Similarity (first 5):\n")
    f.write(np.array2string(tfidf_sim[:5, :5], precision=3))
    f.write("\n\nSVD Similarity (first 5):\n")
    f.write(np.array2string(svd_sim[:5, :5], precision=3))
    f.write("\n\nEmbedding Similarity (first 5):\n")
    f.write(np.array2string(embedding_sim[:5, :5], precision=3))
    
    diff1 = np.abs(tfidf_sim - svd_sim)
    diff2 = np.abs(tfidf_sim - embedding_sim)
    f.write(f"\n\nTF-IDF vs SVD avg diff: {np.mean(diff1):.4f}")
    f.write(f"\nTF-IDF vs Embedding avg diff: {np.mean(diff2):.4f}")

print("similarity_results.txt saved")
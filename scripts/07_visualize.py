import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

output_folder = "../outputs"

with open(os.path.join(output_folder, 'tfidf_matrix.pkl'), 'rb') as f:
    tfidf_matrix = pickle.load(f)

with open(os.path.join(output_folder, 'reduced_matrix.pkl'), 'rb') as f:
    reduced_matrix = pickle.load(f)

with open(os.path.join(output_folder, 'embedding_matrix.pkl'), 'rb') as f:
    embedding_matrix = pickle.load(f)

with open(os.path.join(output_folder, 'cleaned_texts.pkl'), 'rb') as f:
    _, file_names = pickle.load(f)

tfidf_sim = cosine_similarity(tfidf_matrix)
svd_sim = cosine_similarity(reduced_matrix)
embedding_sim = cosine_similarity(embedding_matrix)

diff_tfidf_svd = np.abs(tfidf_sim - svd_sim)
diff_tfidf_embedding = np.abs(tfidf_sim - embedding_sim)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# ردیف اول: ماتریس‌های شباهت
sns.heatmap(tfidf_sim, ax=axes[0, 0], cmap='YlOrRd', cbar_kws={'label': 'Similarity'})
axes[0, 0].set_title('TF-IDF Similarity', fontsize=14)
axes[0, 0].set_xlabel('Ghazal Index')
axes[0, 0].set_ylabel('Ghazal Index')

sns.heatmap(svd_sim, ax=axes[0, 1], cmap='YlOrRd', cbar_kws={'label': 'Similarity'})
axes[0, 1].set_title('SVD Reduced Similarity', fontsize=14)
axes[0, 1].set_xlabel('Ghazal Index')
axes[0, 1].set_ylabel('Ghazal Index')

sns.heatmap(embedding_sim, ax=axes[0, 2], cmap='YlOrRd', cbar_kws={'label': 'Similarity'})
axes[0, 2].set_title('Custom Embedding Similarity', fontsize=14)
axes[0, 2].set_xlabel('Ghazal Index')
axes[0, 2].set_ylabel('Ghazal Index')

# ردیف دوم: تفاوت‌ها
sns.heatmap(diff_tfidf_svd, ax=axes[1, 0], cmap='Blues', cbar_kws={'label': 'Difference'})
axes[1, 0].set_title('TF-IDF vs SVD Difference', fontsize=14)
axes[1, 0].set_xlabel('Ghazal Index')
axes[1, 0].set_ylabel('Ghazal Index')

sns.heatmap(diff_tfidf_embedding, ax=axes[1, 1], cmap='Blues', cbar_kws={'label': 'Difference'})
axes[1, 1].set_title('TF-IDF vs Embedding Difference', fontsize=14)
axes[1, 1].set_xlabel('Ghazal Index')
axes[1, 1].set_ylabel('Ghazal Index')

# نمودار میله‌ای مقایسه‌ی خطاها
avg_diff_svd = np.mean(diff_tfidf_svd)
avg_diff_embedding = np.mean(diff_tfidf_embedding)

axes[1, 2].bar(['SVD vs TF-IDF', 'Embedding vs TF-IDF'], [avg_diff_svd, avg_diff_embedding], color=['#3498db', '#e74c3c'])
axes[1, 2].set_title('Average Information Loss', fontsize=14)
axes[1, 2].set_ylabel('Average Difference')
axes[1, 2].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'similarity_comparison.png'), dpi=300)
plt.show()

print(f"Comparison chart saved to {output_folder}/similarity_comparison.png")
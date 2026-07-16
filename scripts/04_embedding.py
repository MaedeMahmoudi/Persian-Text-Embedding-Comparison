import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

output_folder = "../outputs"

with open(os.path.join(output_folder, 'cleaned_texts.pkl'), 'rb') as f:
    all_ghazals, file_names = pickle.load(f)

vectorizer = TfidfVectorizer(max_features=500, min_df=2, max_df=0.8)
tfidf_matrix = vectorizer.fit_transform(all_ghazals)

svd = TruncatedSVD(n_components=50, random_state=42)
embedding_matrix = svd.fit_transform(tfidf_matrix)
embedding_matrix = normalize(embedding_matrix, norm='l2')

embedding_df = pd.DataFrame(embedding_matrix, index=file_names)
embedding_df.to_csv(os.path.join(output_folder, 'embedding_matrix.csv'), encoding='utf-8-sig')

with open(os.path.join(output_folder, 'embedding_matrix.pkl'), 'wb') as f:
    pickle.dump(embedding_matrix, f)

print(f"Embedding saved: {embedding_matrix.shape}")
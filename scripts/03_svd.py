import os
import pickle
import pandas as pd
from sklearn.decomposition import TruncatedSVD

output_folder = "../outputs"

with open(os.path.join(output_folder, 'tfidf_matrix.pkl'), 'rb') as f:
    tfidf_matrix = pickle.load(f)

svd = TruncatedSVD(n_components=2, random_state=42)
reduced_matrix = svd.fit_transform(tfidf_matrix)

reduced_df = pd.DataFrame(reduced_matrix, columns=['PC1', 'PC2'])
reduced_df.to_csv(os.path.join(output_folder, 'svd_matrix.csv'), index=False)

with open(os.path.join(output_folder, 'reduced_matrix.pkl'), 'wb') as f:
    pickle.dump(reduced_matrix, f)

print(f"SVD saved: {reduced_matrix.shape}")
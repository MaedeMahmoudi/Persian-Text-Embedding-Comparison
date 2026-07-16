import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

output_folder = "../outputs"

with open(os.path.join(output_folder, 'cleaned_texts.pkl'), 'rb') as f:
    all_ghazals, file_names = pickle.load(f)

vectorizer = TfidfVectorizer(max_features=100)
tfidf_matrix = vectorizer.fit_transform(all_ghazals)
feature_names = vectorizer.get_feature_names_out()

tfidf_table = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names,
    index=file_names
)

tfidf_table.to_csv(os.path.join(output_folder, 'tfidf_matrix.csv'), encoding='utf-8-sig')

with open(os.path.join(output_folder, 'tfidf_matrix.pkl'), 'wb') as f:
    pickle.dump(tfidf_matrix, f)

with open(os.path.join(output_folder, 'feature_names.pkl'), 'wb') as f:
    pickle.dump(feature_names, f)

print(f"TF-IDF saved: {tfidf_matrix.shape}")
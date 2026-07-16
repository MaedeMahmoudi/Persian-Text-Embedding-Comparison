import os
import pickle
import torch
import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoTokenizer, AutoModel


model_name = "m3hrdadfi/albert-fa-base-v2"
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


output_folder = "outputs"
with open(os.path.join(output_folder, 'cleaned_texts.pkl'), 'rb') as f:
    all_ghazals, file_names = pickle.load(f)


embeddings = []
for text in all_ghazals:
   
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    embeddings.append(embedding)


embedding_matrix = np.array(embeddings)
embedding_df = pd.DataFrame(embedding_matrix, index=file_names)
embedding_df.to_csv(os.path.join(output_folder, 'albert_embeddings.csv'), encoding='utf-8-sig')

print(f"امبدینگ‌های ALBERT با شکل {embedding_matrix.shape} ذخیره شدند")
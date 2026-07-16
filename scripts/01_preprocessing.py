import os
import re
import pickle
from hazm import Normalizer, word_tokenize

data_folder = "../data/sadi_ghazaliat_DataSet"
output_folder = "../outputs"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

normalizer = Normalizer()

def text_cleaner(raw_text):
    raw_text = normalizer.normalize(raw_text)
    raw_text = re.sub(r'[^\w\s]', '', raw_text)
    tokens = word_tokenize(raw_text)
    return ' '.join(tokens)

all_ghazals = []
file_names = []

for file in sorted(os.listdir(data_folder)):
    if file.endswith('.txt'):
        with open(os.path.join(data_folder, file), 'r', encoding='utf-8') as f:
            original = f.read()
            cleaned = text_cleaner(original)
            if cleaned:
                all_ghazals.append(cleaned)
                file_names.append(file)

print(f"Loaded {len(all_ghazals)} ghazals")

with open(os.path.join(output_folder, 'cleaned_texts.pkl'), 'wb') as f:
    pickle.dump((all_ghazals, file_names), f)

print("cleaned_texts.pkl saved")
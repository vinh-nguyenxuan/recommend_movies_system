import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import ast

# Load the data
data = pd.read_csv(r'C:\Users\TUF\Documents\EXE201\Data\CleanedTMDB1000.csv')
df = data[['genre_ids', 'id', 'title', 'overview', 'keywords', 'cast', 'crew', 'release_date']]

# Clean and process the data
for index, row in df.iterrows():
    tags_sentence = ''.join(row['genre_ids']).replace("'", "")
    df.at[index, 'genre_ids'] = tags_sentence

df.dropna(subset=['overview'], inplace=True)

def converter2(obj):
    M = []
    counter = 0
    for j in ast.literal_eval(obj):
        if counter != 3:
            M.append(j['name'])
            counter += 1
        else:
            break
    return M

def fetch_director(obj):
    N = []
    for k in ast.literal_eval(obj):
        if k['job'] == 'Director':
            N.append(k['name'])
            break
    return N

def converter(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

df['cast'] = df['cast'].apply(converter2)
df['crew'] = df['crew'].apply(fetch_director)
df['keywords'] = df['keywords'].apply(converter)
df['overview'] = df['overview'].apply(lambda x: x.split())
df['keywords'] = df['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
df['cast'] = df['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
df['crew'] = df['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

df['tags'] = df['overview'].apply(str) + df['genre_ids'].apply(str) + df['keywords'].apply(str) + df['cast'].apply(str) + df['crew'].apply(str)

for index, row in df.iterrows():
    tags_sentence = ''.join(row['tags']).replace("'", "").replace(",", " ").replace("[", " ").replace("]", " ")
    df.at[index, 'tags'] = tags_sentence

# Generate embeddings
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
embeddings = model.encode(df['tags'].tolist(), show_progress_bar=True)

# Save the embeddings
np.save('movie_embeddings_multilingual.npy', embeddings)
df.to_csv('processed_data.csv', index=False)

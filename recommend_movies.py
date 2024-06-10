import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import MarianMTModel, MarianTokenizer

# Load the data and embeddings
df = pd.read_csv('processed_data.csv')
embeddings = np.load('movie_embeddings_multilingual.npy')

# Initialize the model for both English and Vietnamese
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# Load translation model and tokenizer
translator_model_name = 'Helsinki-NLP/opus-mt-vi-en'
translator = MarianMTModel.from_pretrained(translator_model_name)
tokenizer = MarianTokenizer.from_pretrained(translator_model_name)

def translate_vi_to_en(text):
    translated = translator.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]

def recommend_movies(prompt_description, model, embeddings, df, top_n=10):
    prompt_description_en = translate_vi_to_en(prompt_description)
    prompt_embedding = model.encode([prompt_description_en])
    cosine_similarities = cosine_similarity(prompt_embedding, embeddings).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    recommended_titles = df['title'].iloc[top_indices]
    return recommended_titles

# Example usage with Vietnamese input
prompt_description = "Một người đàn ông trong bộ đồ sắt cứu thế giới"
recommended_titles = recommend_movies(prompt_description, model, embeddings, df)
print(recommended_titles)

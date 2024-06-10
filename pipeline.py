import argparse
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import MarianMTModel, MarianTokenizer


def translate_vi_to_en(text, translator, tokenizer):
    translated = translator.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]


def recommend_movies(prompt_description, model, embeddings, df, translator, tokenizer, top_n=10):
    # Translate the Vietnamese prompt to English
    prompt_description_en = translate_vi_to_en(prompt_description, translator, tokenizer)

    # Generate embeddings for the prompt
    prompt_embedding = model.encode([prompt_description_en])

    # Calculate cosine similarities between the prompt and movie embeddings
    cosine_similarities = cosine_similarity(prompt_embedding, embeddings).flatten()

    # Get the indices of the top_n most similar movies
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]

    # Get the titles of the recommended movies
    recommended_titles = df['title'].iloc[top_indices]
    return recommended_titles


def main(data_path, user_input, output_file='recommended_movies.txt'):
    # Check if the data file exists
    if not os.path.isfile(data_path):
        print(f"Error: The file {data_path} does not exist.")
        return

    # Load the data and embeddings
    df = pd.read_csv(data_path)
    embeddings_path = 'movie_embeddings_multilingual.npy'
    if not os.path.isfile(embeddings_path):
        print(f"Error: The file {embeddings_path} does not exist.")
        return

    embeddings = np.load(embeddings_path)

    # Initialize the model for both English and Vietnamese
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    # Load translation model and tokenizer
    translator_model_name = 'Helsinki-NLP/opus-mt-vi-en'
    translator = MarianMTModel.from_pretrained(translator_model_name)
    tokenizer = MarianTokenizer.from_pretrained(translator_model_name)

    # Get recommendations
    recommended_titles = recommend_movies(user_input, model, embeddings, df, translator, tokenizer)

    # Save recommendations to a file
    with open(output_file, 'w', encoding='utf-8') as f:
        for title in recommended_titles:
            f.write(f"{title}\n")

    print(f"Recommended movies have been saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Movie Recommendation Pipeline')
    parser.add_argument('--data', type=str, required=True, help='Path to the TMDB data CSV file')
    parser.add_argument('--input', type=str, required=True, help='User input description for movie recommendation')
    args = parser.parse_args()

    main(args.data, args.input)

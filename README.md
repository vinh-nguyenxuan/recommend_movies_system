# Movie Recommendation System

This project provides a pipeline to recommend movies based on user descriptions using a dataset of movies and sentence embeddings.

## Steps to Follow

1. **Upload Data**

   Download the dataset from [Kaggle](https://www.kaggle.com/datasets/apkaayush/tmdb-10000-movies-dataset).

2. **Install Requirements**

   Make sure you have Python installed. Then, install the necessary packages using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   python train_model.py
   python recommend_movies.py
   python pipeline.py --data path/to/tmdb.csv --input "Một người đàn ông sử dụng sức mạnh của nhện để cứu thế giới"

# Movie Recommendation System

This project builds a movie recommendation system using the provided `movies.csv` and `ratings.csv` datasets.

## Dataset

The project uses two files:

- `movies.csv`: contains movie information:
  - `movieId`
  - `title`
  - `genres`

- `ratings.csv`: contains user ratings:
  - `userId`
  - `movieId`
  - `rating`
  - `timestamp`

## Project idea

Recommendation systems are used in platforms like Netflix, Spotify, Amazon, YouTube, and online shopping apps.  
The goal of this project is to recommend movies to users based on their past ratings and movie similarities.

## Methods used

### 1. Popularity-based recommendation

This recommends globally popular movies using:

- average rating
- number of ratings
- weighted score

This is useful for new users who do not have rating history.

### 2. Content-based filtering

This recommends movies similar to a selected movie based on movie genres.

Example:

If the user likes `Toy Story`, the system searches for movies with similar genres such as:

- Adventure
- Animation
- Children
- Comedy
- Fantasy

### 3. Collaborative filtering

This recommends movies based on user behavior.

The idea is:

> If users who liked Movie A also liked Movie B, then Movie B can be recommended to another user who liked Movie A.

The implementation uses item-item KNN with cosine similarity.

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app

```bash
streamlit run app.py
```

### 3. Run the recommender in terminal

```bash
python recommender.py
```

## Project structure

```text
movie_recommender_project/
│
├── app.py
├── recommender.py
├── requirements.txt
├── README.md
└── data/
    ├── movies.csv
    └── ratings.csv
```

## Possible improvements

- Add model evaluation with train/test split
- Add matrix factorization using SVD
- Add user login and favorite list
- Add poster images using an external movie API
- Deploy the app with Streamlit Cloud or Hugging Face Spaces
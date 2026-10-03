from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class MovieRecommender:
    """
    Movie recommendation system using:
    1) Popularity-based recommendation
    2) Content-based filtering using movie genres
    3) Collaborative filtering using item-item KNN on user ratings
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.movies = pd.read_csv(self.data_dir / "movies.csv")
        self.ratings = pd.read_csv(self.data_dir / "ratings.csv")

        self._prepare_data()
        self._build_popularity_model()
        self._build_content_model()
        self._build_collaborative_model()

    def _prepare_data(self):
        self.movies["genres"] = self.movies["genres"].fillna("")
        self.movies["genres_clean"] = (
            self.movies["genres"]
            .str.replace("|", " ", regex=False)
            .str.replace("(no genres listed)", "", regex=False)
        )
        self.movies["title_lower"] = self.movies["title"].str.lower()

        self.ratings = self.ratings.dropna(subset=["userId", "movieId", "rating"])
        self.ratings["userId"] = self.ratings["userId"].astype(int)
        self.ratings["movieId"] = self.ratings["movieId"].astype(int)

    def _build_popularity_model(self):
        stats = (
            self.ratings
            .groupby("movieId")
            .agg(avg_rating=("rating", "mean"), rating_count=("rating", "count"))
            .reset_index()
        )

        C = stats["avg_rating"].mean()
        m = stats["rating_count"].quantile(0.75)

        stats["weighted_score"] = (
            (stats["rating_count"] / (stats["rating_count"] + m)) * stats["avg_rating"]
            + (m / (stats["rating_count"] + m)) * C
        )

        self.popularity_table = (
            stats.merge(self.movies[["movieId", "title", "genres"]], on="movieId", how="left")
            .sort_values("weighted_score", ascending=False)
            .reset_index(drop=True)
        )

    def _build_content_model(self):
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.content_matrix = self.tfidf.fit_transform(self.movies["genres_clean"])

        self.content_knn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.content_knn.fit(self.content_matrix)

        self.movie_id_to_index = pd.Series(
            self.movies.index.values,
            index=self.movies["movieId"]
        ).to_dict()

    def _build_collaborative_model(self):
        self.user_movie_matrix = self.ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating"
        ).fillna(0)

        self.movie_ids_cf = list(self.user_movie_matrix.columns)
        self.movie_id_to_cf_index = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids_cf)}

        item_user_matrix = csr_matrix(self.user_movie_matrix.T.values)

        self.cf_knn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.cf_knn.fit(item_user_matrix)
        self.item_user_matrix = item_user_matrix

    def search_movies(self, text: str, limit: int = 10) -> pd.DataFrame:
        text = text.lower().strip()
        result = self.movies[self.movies["title_lower"].str.contains(text, na=False)]
        return result[["movieId", "title", "genres"]].head(limit).reset_index(drop=True)

    def popular_movies(self, n: int = 10, min_ratings: int = 20) -> pd.DataFrame:
        result = self.popularity_table[self.popularity_table["rating_count"] >= min_ratings]
        return result[["title", "genres", "avg_rating", "rating_count", "weighted_score"]].head(n)

    def recommend_by_content(self, movie_title: str, n: int = 10) -> pd.DataFrame:
        matches = self.search_movies(movie_title, limit=1)
        if matches.empty:
            return pd.DataFrame(columns=["title", "genres", "similarity"])

        movie_id = int(matches.iloc[0]["movieId"])
        movie_index = self.movie_id_to_index[movie_id]

        distances, indices = self.content_knn.kneighbors(
            self.content_matrix[movie_index],
            n_neighbors=min(n + 1, len(self.movies))
        )

        rows = []
        for distance, index in zip(distances.flatten(), indices.flatten()):
            if index == movie_index:
                continue
            similarity = 1 - distance
            rows.append({
                "title": self.movies.iloc[index]["title"],
                "genres": self.movies.iloc[index]["genres"],
                "similarity": round(float(similarity), 4)
            })

        return pd.DataFrame(rows).head(n)

    def similar_movies_collaborative(self, movie_title: str, n: int = 10) -> pd.DataFrame:
        matches = self.search_movies(movie_title, limit=1)
        if matches.empty:
            return pd.DataFrame(columns=["title", "genres", "similarity"])

        movie_id = int(matches.iloc[0]["movieId"])
        if movie_id not in self.movie_id_to_cf_index:
            return pd.DataFrame(columns=["title", "genres", "similarity"])

        cf_index = self.movie_id_to_cf_index[movie_id]

        distances, indices = self.cf_knn.kneighbors(
            self.item_user_matrix[cf_index],
            n_neighbors=min(n + 1, len(self.movie_ids_cf))
        )

        rows = []
        for distance, idx in zip(distances.flatten(), indices.flatten()):
            similar_movie_id = self.movie_ids_cf[idx]
            if similar_movie_id == movie_id:
                continue

            movie_info = self.movies[self.movies["movieId"] == similar_movie_id]
            if movie_info.empty:
                continue

            rows.append({
                "title": movie_info.iloc[0]["title"],
                "genres": movie_info.iloc[0]["genres"],
                "similarity": round(float(1 - distance), 4)
            })

        return pd.DataFrame(rows).head(n)

    def recommend_for_user(self, user_id: int, n: int = 10, min_rating: float = 4.0) -> pd.DataFrame:
        """
        Recommend unseen movies to a user based on movies they rated highly.
        """
        user_ratings = self.ratings[self.ratings["userId"] == int(user_id)]

        if user_ratings.empty:
            return self.popular_movies(n=n)

        seen_movie_ids = set(user_ratings["movieId"].unique())
        liked_movies = user_ratings[user_ratings["rating"] >= min_rating]

        if liked_movies.empty:
            return self.popular_movies(n=n)

        scores = {}

        for _, row in liked_movies.iterrows():
            movie_id = int(row["movieId"])
            user_rating = float(row["rating"])

            if movie_id not in self.movie_id_to_cf_index:
                continue

            cf_index = self.movie_id_to_cf_index[movie_id]
            distances, indices = self.cf_knn.kneighbors(
                self.item_user_matrix[cf_index],
                n_neighbors=min(30, len(self.movie_ids_cf))
            )

            for distance, idx in zip(distances.flatten(), indices.flatten()):
                candidate_movie_id = self.movie_ids_cf[idx]

                if candidate_movie_id in seen_movie_ids or candidate_movie_id == movie_id:
                    continue

                similarity = 1 - distance
                scores[candidate_movie_id] = scores.get(candidate_movie_id, 0) + similarity * user_rating

        if not scores:
            return self.popular_movies(n=n)

        recommendations = (
            pd.DataFrame(
                [{"movieId": movie_id, "score": score} for movie_id, score in scores.items()]
            )
            .merge(self.movies[["movieId", "title", "genres"]], on="movieId", how="left")
            .sort_values("score", ascending=False)
            .head(n)
        )

        return recommendations[["title", "genres", "score"]].reset_index(drop=True)

    def user_profile(self, user_id: int, n: int = 10) -> pd.DataFrame:
        profile = (
            self.ratings[self.ratings["userId"] == int(user_id)]
            .merge(self.movies[["movieId", "title", "genres"]], on="movieId", how="left")
            .sort_values("rating", ascending=False)
            .head(n)
        )
        return profile[["title", "genres", "rating"]].reset_index(drop=True)


if __name__ == "__main__":
    recommender = MovieRecommender(data_dir="data")

    print("\nTop popular movies:")
    print(recommender.popular_movies(n=5))

    print("\nMovies similar to Toy Story:")
    print(recommender.recommend_by_content("Toy Story", n=5))

    print("\nRecommendations for user 1:")
    print(recommender.recommend_for_user(user_id=1, n=5))
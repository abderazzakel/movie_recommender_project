import streamlit as st
import pandas as pd

from recommender import MovieRecommender


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


@st.cache_resource
def load_recommender():
    return MovieRecommender(data_dir="data")


recommender = load_recommender()

st.title("🎬 Movie Recommendation System")
st.write(
    "A simple recommendation system using popularity-based filtering, "
    "content-based filtering, and collaborative filtering."
)

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a module",
    [
        "Recommend movies for a user",
        "Find similar movies",
        "Top popular movies",
        "Dataset overview"
    ]
)

if page == "Recommend movies for a user":
    st.subheader("👤 Personalized recommendations")

    user_ids = sorted(recommender.ratings["userId"].unique())
    user_id = st.selectbox("Select a user", user_ids)
    n = st.slider("Number of recommendations", 5, 30, 10)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### User's highest-rated movies")
        st.dataframe(recommender.user_profile(user_id, n=10), use_container_width=True)

    with col2:
        st.markdown("### Recommended movies")
        st.dataframe(recommender.recommend_for_user(user_id, n=n), use_container_width=True)

elif page == "Find similar movies":
    st.subheader("🔎 Similar movie search")

    search_text = st.text_input("Search a movie title", value="Toy Story")
    matches = recommender.search_movies(search_text, limit=20)

    if matches.empty:
        st.warning("No movie found. Try another title.")
    else:
        movie_title = st.selectbox("Choose a movie", matches["title"].tolist())
        method = st.radio(
            "Recommendation method",
            ["Content-based filtering", "Collaborative filtering"]
        )
        n = st.slider("Number of similar movies", 5, 30, 10)

        if method == "Content-based filtering":
            result = recommender.recommend_by_content(movie_title, n=n)
        else:
            result = recommender.similar_movies_collaborative(movie_title, n=n)

        st.dataframe(result, use_container_width=True)

elif page == "Top popular movies":
    st.subheader("⭐ Popular movies")
    n = st.slider("Number of movies", 5, 50, 10)
    min_ratings = st.slider("Minimum number of ratings", 1, 100, 20)

    result = recommender.popular_movies(n=n, min_ratings=min_ratings)
    st.dataframe(result, use_container_width=True)

elif page == "Dataset overview":
    st.subheader("📊 Dataset overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Movies", recommender.movies["movieId"].nunique())
    col2.metric("Users", recommender.ratings["userId"].nunique())
    col3.metric("Ratings", len(recommender.ratings))
    col4.metric("Average rating", round(recommender.ratings["rating"].mean(), 2))

    st.markdown("### Movies data")
    st.dataframe(recommender.movies[["movieId", "title", "genres"]].head(20), use_container_width=True)

    st.markdown("### Ratings data")
    st.dataframe(recommender.ratings.head(20), use_container_width=True)
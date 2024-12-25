from typing import List, Dict, Any
import pandas as pd
from ..models.movie import Movie

class MovieQueryEngine:
    def __init__(self, vector_store_engine, movie_data: pd.DataFrame):
        self.engine = vector_store_engine
        self.movie_data = movie_data

    def enhance_query(self, query: str) -> str:
        """Enhance the user query to get better recommendations."""
        return f"""
        Based on the following user request, recommend movies and explain why they match:
        {query}
        
        For each recommended movie, provide:
        1. Title and year
        2. Genres
        3. Average rating and number of votes
        4. A brief explanation of why it matches the request
        5. Any notable aspects (high budget, part of a collection, etc.)
        
        Focus on movies that best match the user's specific preferences and requirements.
        """

    def process_similar_movies_query(self, movie_title: str) -> str:
        """Create a query to find movies similar to a given title."""
        movie = self.movie_data[
            self.movie_data['original_title'].str.lower() == movie_title.lower()
        ]
        
        if len(movie) == 0:
            return None
            
        movie = movie.iloc[0]
        query = f"""
        Find movies similar to '{movie_title}' with these characteristics:
        - Genres: {', '.join(movie['genres'])}
        - Runtime: {movie['runtime']} minutes
        - Rating: {movie['vote_average']}
        
        Consider both the plot elements and these characteristics.
        Prioritize movies with similar genres and themes.
        """
        return query

    def filter_recommendations(self, 
                             response: str, 
                             min_rating: float = 0.0,
                             max_budget: float = float('inf'),
                             genres: List[str] = None) -> str:
        """
        Filter and enhance the recommendation response based on criteria.
        """
        movies = self.movie_data[
            (self.movie_data['vote_average'] >= min_rating) &
            (self.movie_data['budget'] <= max_budget)
        ]
        
        if genres:
            movies = movies[
                movies['genres'].apply(lambda x: any(genre in x for genre in genres))
            ]
        
        results = []
        for _, movie in movies.iterrows():
            results.append(
                f"{movie['original_title']} ({', '.join(movie['genres'])})"
                f"\nRating: {movie['vote_average']}/10 ({movie['vote_count']} votes)"
                f"\nBudget: ${movie['budget']:,.2f}"
            )
            
        return "\n\n".join(results)

    def format_response(self, response: str) -> str:
        """Format the recommendation response for better readability."""
        formatted = "🎬 Movie Recommendations:\n\n"
        formatted += response
        formatted += "\n\n💡 Note: Ratings are out of 10, based on user votes."
        return formatted
from typing import List, Dict, Any
import pandas as pd
from llama_index import Document

from ..config.settings import TOP_K_RECOMMENDATIONS
from ..indexing.vector_store import MovieVectorStore
from .query_engine import MovieQueryEngine

class MovieRecommendationBot:
    def __init__(
        self,
        documents: List[Document],
        movie_data: pd.DataFrame,
        azure_credentials: Dict[str, str],
        top_k: int = TOP_K_RECOMMENDATIONS
    ):
        """
        Initialize the movie recommendation chatbot.
        
        Args:
            documents: List of processed Document objects
            movie_data: DataFrame containing movie information
            azure_credentials: Dictionary containing Azure OpenAI credentials
            top_k: Number of recommendations to return
        """
        self.movie_data = movie_data
        self.top_k = top_k
        
        self.vector_store = MovieVectorStore()
        self.vector_store.initialize_index(documents)
        
        vector_engine = self.vector_store.get_query_engine(top_k=top_k)
        self.query_engine = MovieQueryEngine(vector_engine, movie_data)

    def get_recommendation(self, query: str) -> str:
        """
        Get movie recommendations based on the user query.
        
        Args:
            query: User's request for movie recommendations
            
        Returns:
            Formatted response with movie recommendations
        """
        enhanced_query = self.query_engine.enhance_query(query)
        response = self.query_engine.engine.query(enhanced_query)
        formatted_response = self.query_engine.format_response(response.response)
        
        return formatted_response

    def get_similar_movies(self, movie_title: str) -> str:
        """
        Find movies similar to a given movie title.
        
        Args:
            movie_title: Title of the movie to find similarities for
            
        Returns:
            Formatted response with similar movies
        """
        query = self.query_engine.process_similar_movies_query(movie_title)
        
        if query is None:
            return f"Sorry, I couldn't find the movie '{movie_title}' in my database."
        
        response = self.query_engine.engine.query(query)
        formatted_response = self.query_engine.format_response(response.response)
        
        return formatted_response

    def filter_recommendations(self, 
                             query: str,
                             min_rating: float = 0.0,
                             max_budget: float = float('inf'),
                             genres: List[str] = None) -> str:
        """
        Get filtered movie recommendations.
        
        Args:
            query: Base query for recommendations
            min_rating: Minimum rating threshold
            max_budget: Maximum budget threshold
            genres: List of required genres
            
        Returns:
            Filtered and formatted recommendations
        """
        response = self.get_recommendation(query)
        
        filtered_response = self.query_engine.filter_recommendations(
            response,
            min_rating=min_rating,
            max_budget=max_budget,
            genres=genres
        )
        
        return filtered_response
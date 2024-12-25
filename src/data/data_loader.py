import pandas as pd
from pathlib import Path
from typing import Tuple

from ..config.settings import RAW_DATA_DIR, MIN_BUDGET_FILTER
from .preprocessor import MoviePreprocessor

class MovieDataLoader:
    def __init__(self, data_file: str = "movies_metadata.csv"):
        self.data_path = Path(RAW_DATA_DIR) / data_file
        self.preprocessor = MoviePreprocessor()

    def load_and_preprocess(self) -> Tuple[pd.DataFrame, list]:
        """
        Load and preprocess the movie dataset.
        
        Returns:
            Tuple containing:
            - Processed DataFrame
            - List of Document objects ready for indexing
        """
        df = pd.read_csv(self.data_path)
        df = df.apply(self.preprocessor.preprocess_row, axis=1)
        
        df = df[df['budget'] > MIN_BUDGET_FILTER]
        
        df = df[[
            'adult',
            'belongs_to_collection',
            'budget',
            'genres',
            'original_language',
            'original_title',
            'overview',
            'popularity',
            'revenue',
            'runtime',
            'vote_average',
            'vote_count'
        ]]
        
        documents = self.preprocessor.create_documents(df)
        return df, documents
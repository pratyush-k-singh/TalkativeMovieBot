import pandas as pd
from typing import List
from llama_index import Document

class MoviePreprocessor:
    def preprocess_row(self, row: pd.Series) -> pd.Series:
        """Process a single row of movie data."""
        belongs_to_collection = row['belongs_to_collection']
        belongs_to_collection = 'NULL' if pd.isnull(belongs_to_collection) else belongs_to_collection
        belongs_to_collection = eval(belongs_to_collection)['name'] if belongs_to_collection != 'NULL' else 'NULL'

        genres = row['genres']
        genres = 'NULL' if pd.isnull(genres) else genres
        if genres != 'NULL':
            genres = eval(genres)
            genres = [genre['name'] for genre in genres]
            
        row['belongs_to_collection'] = belongs_to_collection
        row['genres'] = genres

        return row

    def create_documents(self, df: pd.DataFrame) -> List[Document]:
        """Create Document objects from preprocessed DataFrame."""
        documents = []
        for i, row in df.iterrows():
            doc = Document(
                id=str(i),
                text=row['overview'],
                metadata={
                    'title': row['original_title'],
                    'genres': row['genres'],
                    'belongs_to_collection': row['belongs_to_collection'],
                    'budget': float(row['budget']),
                    'popularity': float(row['popularity']),
                    'revenue': float(row['revenue']),
                    'runtime': float(row['runtime']) if pd.notnull(row['runtime']) else 0.0,
                    'vote_average': float(row['vote_average']),
                    'vote_count': float(row['vote_count'])
                }
            )
            documents.append(doc)
        return documents
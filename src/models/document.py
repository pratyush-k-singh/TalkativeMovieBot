from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime

@dataclass
class MovieDocument:
    """
    Represents a movie document with metadata for vector storage and retrieval.
    Provides a standardized way to handle movie data throughout the application.
    """
    id: str
    text: str  # The movie overview/description
    metadata: Dict[str, Any]
    embedding: Optional[list] = None
    timestamp: datetime = datetime.now()

    def to_llama_doc(self) -> Dict[str, Any]:
        """Convert to LlamaIndex Document format."""
        return {
            "id_": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_llama_doc(cls, doc: Dict[str, Any]) -> 'MovieDocument':
        """Create MovieDocument from LlamaIndex Document."""
        return cls(
            id=doc.get("id_", ""),
            text=doc.get("text", ""),
            metadata=doc.get("metadata", {}),
            embedding=doc.get("embedding")
        )
    
    def update_metadata(self, new_metadata: Dict[str, Any]) -> None:
        """Update document metadata while preserving existing values."""
        self.metadata.update(new_metadata)
    
    def get_genre_str(self) -> str:
        """Get genres as a comma-separated string."""
        genres = self.metadata.get('genres', [])
        if isinstance(genres, list):
            return ', '.join(genres)
        return str(genres)
    
    def get_collection(self) -> Optional[str]:
        """Get movie collection name if it exists."""
        collection = self.metadata.get('belongs_to_collection')
        return None if collection == 'NULL' else collection
    
    def get_metrics(self) -> Dict[str, float]:
        """Get numerical metrics for the movie."""
        return {
            'budget': float(self.metadata.get('budget', 0)),
            'revenue': float(self.metadata.get('revenue', 0)),
            'runtime': float(self.metadata.get('runtime', 0)),
            'vote_average': float(self.metadata.get('vote_average', 0)),
            'vote_count': float(self.metadata.get('vote_count', 0)),
            'popularity': float(self.metadata.get('popularity', 0))
        }
    
    def calculate_engagement_score(self) -> float:
        """Calculate an engagement score based on votes and popularity."""
        metrics = self.get_metrics()
        vote_weight = min(metrics['vote_count'] / 1000, 1.0)  # Normalize vote count
        return (metrics['vote_average'] * vote_weight + metrics['popularity']) / 2
    
    def calculate_roi(self) -> float:
        """Calculate return on investment."""
        metrics = self.get_metrics()
        if metrics['budget'] == 0:
            return 0.0
        return (metrics['revenue'] - metrics['budget']) / metrics['budget']
    
    def is_successful(self) -> bool:
        """Determine if the movie was successful based on ROI and engagement."""
        roi = self.calculate_roi()
        engagement = self.calculate_engagement_score()
        return roi > 0.5 and engagement > 5.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format with computed metrics."""
        return {
            'id': self.id,
            'text': self.text,
            'metadata': self.metadata,
            'genres': self.get_genre_str(),
            'collection': self.get_collection(),
            'metrics': self.get_metrics(),
            'engagement_score': self.calculate_engagement_score(),
            'roi': self.calculate_roi(),
            'is_successful': self.is_successful(),
            'timestamp': self.timestamp.isoformat()
        }
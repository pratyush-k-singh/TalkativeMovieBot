from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class Movie:
    id: str
    title: str
    genres: List[str]
    overview: str
    budget: float
    revenue: float
    runtime: float
    vote_average: float
    vote_count: float
    popularity: float
    collection: Optional[str] = None
    original_language: str = "en"
    adult: bool = False

    @property
    def roi(self) -> float:
        """Calculate Return on Investment"""
        if self.budget == 0:
            return 0.0
        return (self.revenue - self.budget) / self.budget

    @property
    def popularity_score(self) -> float:
        """Calculate normalized popularity score"""
        vote_weight = min(self.vote_count / 1000, 1.0)
        return (self.vote_average * vote_weight + self.popularity) / 2

    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            'id': self.id,
            'title': self.title,
            'genres': self.genres,
            'overview': self.overview,
            'budget': self.budget,
            'revenue': self.revenue,
            'runtime': self.runtime,
            'vote_average': self.vote_average,
            'vote_count': self.vote_count,
            'popularity': self.popularity,
            'collection': self.collection,
            'original_language': self.original_language,
            'adult': self.adult,
            'roi': self.roi,
            'popularity_score': self.popularity_score
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Movie':
        """Create Movie instance from dictionary"""
        return cls(
            id=data['id'],
            title=data['title'],
            genres=data['genres'],
            overview=data['overview'],
            budget=data['budget'],
            revenue=data['revenue'],
            runtime=data['runtime'],
            vote_average=data['vote_average'],
            vote_count=data['vote_count'],
            popularity=data['popularity'],
            collection=data.get('collection'),
            original_language=data.get('original_language', 'en'),
            adult=data.get('adult', False)
        )
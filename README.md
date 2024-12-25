# Movie Recommendation Chatbot

An advanced movie recommendation system leveraging LlamaIndex, FAISS vector search, and Azure OpenAI for intelligent, context-aware movie suggestions. This system combines semantic search capabilities with natural language understanding to provide personalized movie recommendations.

## 🎯 Key Features

- **Semantic Search**: Uses FAISS vector store for efficient similarity search
- **Natural Language Understanding**: Powered by Azure OpenAI
- **Advanced Filtering**: Multi-dimensional filtering by genre, budget, ratings, etc.
- **Performance Optimized**: 
  - Caching mechanisms for frequent queries
  - Batch processing for vector operations
  - Efficient index management
  - Approximate nearest neighbor search options
- **Persistent Storage**: Maintains vector indices and metadata across sessions

## 🧠 Technical Architecture

### Vector Store (FAISS Integration)
- **Index Types**:
  - Exact Search: `IndexFlatIP` for precise recommendations
  - Approximate Search: `IndexIVFFlat` for faster, approximate matching
- **Optimization Features**:
  - Document lookup cache
  - Batched index updates
  - Query result caching
  - Automatic index optimization

### Query Engine
- **Query Enhancement**:
  - Template-based query expansion
  - Context-aware query modification
  - Genre embedding precomputation
- **Results Processing**:
  - Efficient batch processing
  - Cached similarity computations
  - Smart ranking algorithms

### Data Processing
- **Preprocessing Pipeline**:
  - Genre normalization
  - Collection handling
  - Metadata enrichment
- **Document Management**:
  - Efficient document storage
  - Batch update capabilities
  - Metadata indexing

## 🚀 Performance Optimizations

### Vector Search
- Uses FAISS's `IndexIVFFlat` for approximate search when speed is priority
- Implements query vector caching for frequent searches
- Batches similar movie lookups for efficiency

### Query Processing
- Caches enhanced queries with TTL
- Precomputes genre embeddings
- Implements LRU cache for query vectors
- Uses heap-based priority queue for efficient top-K selection

### Memory Management
- Efficient document lookup with dictionary storage
- Batched processing for large datasets
- Automatic cleanup of expired cache entries

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd movie-recommender
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Azure OpenAI credentials:
```bash
cp .env.example .env
# Edit .env with your credentials
```

5. Prepare your data:
- Place your movie dataset in `data/raw/movies_metadata.csv`
- Run initial preprocessing: `python scripts/prepare_data.py`

## 💻 Usage

### Basic Usage
```python
from movie_recommender import MovieRecommendationBot

# Initialize the bot
bot = MovieRecommendationBot(
    data_path='data/raw/movies_metadata.csv',
    azure_credentials=your_credentials
)

# Get recommendations
recommendations = bot.get_recommendation(
    "I want an action movie with high ratings and a budget under 50 million"
)

# Find similar movies
similar_movies = bot.get_similar_movies("The Dark Knight")
```

### Advanced Features

#### Custom Filtering
```python
recommendations = bot.get_recommendation(
    query="Show me sci-fi movies",
    min_rating=7.0,
    max_budget=100000000,
    genres=['Science Fiction', 'Action']
)
```

#### Performance Tuning
```python
bot = MovieRecommendationBot(
    use_approximate_search=True,  # Faster but slightly less accurate
    cache_size=2000,             # Increase cache size for better performance
    batch_size=50               # Adjust batch size for your needs
)
```

## 🔧 Configuration

Key configuration options in `config/settings.py`:
```python
# Vector store settings
EMBEDDING_DIMENSION = 384
USE_APPROXIMATE_SEARCH = False
CACHE_SIZE = 1000

# Query engine settings
TOP_K_RECOMMENDATIONS = 3
MIN_BUDGET_FILTER = 1_000_000

# Performance settings
BATCH_SIZE = 100
CACHE_TTL_MINUTES = 60
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## 📈 Performance Metrics

- Query response time: ~100-200ms (cached)
- Vector search time: ~50ms (approximate), ~200ms (exact)
- Memory usage: ~500MB base, scales with dataset size
- Cache hit ratio: ~70% (typical usage)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FAISS by Facebook Research
- LlamaIndex for vector store integration
- Azure OpenAI for language models
- The Movie Database (TMDb) for movie data
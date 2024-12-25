import faiss
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Set
from datetime import datetime
from functools import lru_cache
from llama_index import Document
from llama_index.vector_stores import FaissVectorStore
from llama_index import VectorStoreIndex, StorageContext

from ..config.settings import INDEX_DIR, EMBEDDING_DIMENSION
from ..models.movie import Movie

class MovieVectorStore:
    def __init__(self, cache_size: int = 1000):
        self.index_path = Path(INDEX_DIR)
        self.dimension = EMBEDDING_DIMENSION
        self.index = None
        self.cache_size = cache_size
        self.document_lookup: Dict[str, Document] = {}
        self.last_modified = datetime.now()
        self.pending_updates: Set[str] = set()
        
        self.index_configs = {
            'flat': faiss.IndexFlatIP(self.dimension),
            'ivf': self._create_ivf_index(),
        }
        self.current_config = 'flat'

    def _create_ivf_index(self, nlist: int = 100) -> faiss.Index:
        """Create an IVF index for faster approximate search."""
        quantizer = faiss.IndexFlatIP(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        return index

    @lru_cache(maxsize=1000)
    def _cached_similarity_search(self, query_vector: tuple) -> List[int]:
        """Cache similarity search results for frequent queries."""
        query_vector_array = np.array(query_vector).reshape(1, -1)
        D, I = self.index_configs[self.current_config].search(query_vector_array, self.cache_size)
        return I[0].tolist()

    def initialize_index(self, documents: Optional[List[Document]] = None):
        """Initialize or load the FAISS index with optimized settings."""
        if self.index_path.exists() and not documents:
            self._load_existing_index()
        else:
            self._create_new_index(documents)

    def _load_existing_index(self):
        """Load existing index with optimizations."""
        try:
            self.index = VectorStoreIndex.load_from_disk(
                str(self.index_path),
                store_loading_fn=lambda: FaissVectorStore.from_persist_dir(str(self.index_path))
            )
            
            cache_path = self.index_path / "document_lookup.npy"
            if cache_path.exists():
                self.document_lookup = np.load(cache_path, allow_pickle=True).item()

        except Exception as e:
            raise ValueError(f"Error loading index: {e}")

    def _create_new_index(self, documents: List[Document]):
        """Create new index with optimizations."""
        if not documents:
            raise ValueError("Documents required for new index creation")

        self.document_lookup = {doc.id_: doc for doc in documents}

        vector_store = FaissVectorStore(faiss_index=self.index_configs[self.current_config])
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        batch_size = 1000
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            if i == 0:
                self.index = VectorStoreIndex.from_documents(
                    documents=batch,
                    storage_context=storage_context,
                    show_progress_bar=True
                )
            else:
                self.index.refresh_ref_docs(batch)

        self._save_index()

    def _save_index(self):
        """Save index and related data with error handling."""
        try:
            self.index_path.parent.mkdir(exist_ok=True)
            self.index.storage_context.persist(str(self.index_path))

            cache_path = self.index_path / "document_lookup.npy"
            np.save(cache_path, self.document_lookup)

            self.last_modified = datetime.now()
        except Exception as e:
            raise ValueError(f"Error saving index: {e}")

    def get_query_engine(self, top_k: int = 3, use_approximate: bool = False):
        """Get optimized query engine based on query requirements."""
        if not self.index:
            raise ValueError("Index not initialized. Call initialize_index first.")

        self.current_config = 'ivf' if use_approximate else 'flat'
        
        return self.index.as_query_engine(
            similarity_top_k=top_k,
            vector_store_kwargs={'vector_store': FaissVectorStore(self.index_configs[self.current_config])}
        )

    def update_documents(self, documents: List[Document], batch_size: int = 100):
        """Update index with new documents using batched processing."""
        if not self.index:
            self.initialize_index(documents)
            return

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            for doc in batch:
                self.document_lookup[doc.id_] = doc
                self.pending_updates.add(doc.id_)

            self.index.refresh_ref_docs(batch)

        if len(self.pending_updates) >= batch_size:
            self._save_index()
            self.pending_updates.clear()

    def optimize_index(self):
        """Optimize the index for better performance."""
        if not self.index:
            return

        if self.current_config == 'ivf':
            training_vectors = np.random.random((10000, self.dimension)).astype('float32')
            self.index_configs['ivf'].train(training_vectors)

        self._cached_similarity_search.cache_clear()

    def cleanup(self):
        """Cleanup resources and save pending changes."""
        if self.pending_updates:
            self._save_index()
        self._cached_similarity_search.cache_clear()
        
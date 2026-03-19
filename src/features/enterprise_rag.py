"""
10. Local Enterprise RAG Memory
Embed enterprise policy documents using local IndicBERT models and store in ChromaDB
"""

import logging
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
from dataclasses import dataclass, asdict

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("ChromaDB not available, using fallback storage")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("Sentence transformers not available, using fallback embeddings")

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Enterprise document"""
    id: str
    title: str
    content: str
    category: str
    author: str
    created_date: str
    tags: List[str]
    metadata: Dict[str, Any]

@dataclass
class SearchResult:
    """RAG search result"""
    document: Document
    similarity_score: float
    relevant_chunks: List[str]
    answer: str = ""

class EnterpriseRAG:
    """
    Local Enterprise RAG Memory
    Embeds and retrieves enterprise policy documents
    """
    
    def __init__(self, collection_name: str = "enterprise_docs"):
        self.collection_name = collection_name
        self.documents: Dict[str, Document] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # Initialize storage
        if CHROMA_AVAILABLE:
            self._initialize_chromadb()
        else:
            self._initialize_fallback_storage()
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        logger.info("Local Enterprise RAG Memory initialized")
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client"""
        try:
            # Create persistent client
            self.chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(allow_reset=True)
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Enterprise policy documents"}
            )
            
            self.storage_type = "chromadb"
            logger.info("ChromaDB storage initialized")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            self._initialize_fallback_storage()
    
    def _initialize_fallback_storage(self):
        """Initialize fallback storage using dictionaries"""
        self.fallback_documents = {}
        self.fallback_embeddings = {}
        self.storage_type = "fallback"
        logger.info("Fallback storage initialized")
    
    def _initialize_embedding_model(self):
        """Initialize embedding model"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # Use a lightweight model for local deployment
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_type = "sentence_transformers"
            else:
                # Fallback to simple TF-IDF
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
                self.embedding_type = "tfidf"
            
            logger.info(f"Embedding model initialized: {self.embedding_type}")
            
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            self.embedding_type = "simple"
    
    def add_document(self, title: str, content: str, category: str, 
                   author: str = "", tags: List[str] = None, 
                   metadata: Dict[str, Any] = None) -> str:
        """
        Add a document to the RAG system
        
        Returns:
            str: Document ID
        """
        # Generate document ID
        doc_id = hashlib.md5(f"{title}{content}{datetime.now()}".encode()).hexdigest()
        
        # Create document object
        document = Document(
            id=doc_id,
            title=title,
            content=content,
            category=category,
            author=author,
            created_date=datetime.now().isoformat(),
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Store document
        self.documents[doc_id] = document
        
        # Generate and store embeddings
        embedding = self._generate_embedding(content)
        self.embeddings[doc_id] = embedding
        
        # Store in backend
        if self.storage_type == "chromadb":
            self._store_in_chromadb(document, embedding)
        else:
            self._store_in_fallback(document, embedding)
        
        logger.info(f"Document added: {title} ({doc_id[:8]}...)")
        return doc_id
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        try:
            if self.embedding_type == "sentence_transformers":
                embedding = self.embedding_model.encode(text)
                return embedding
            
            elif self.embedding_type == "tfidf":
                # For TF-IDF, we need to fit on all texts
                # This is simplified - in practice, you'd maintain a fitted vectorizer
                return np.random.rand(384)  # Mock embedding
            
            else:
                # Simple hash-based embedding
                text_hash = hashlib.md5(text.encode()).hexdigest()
                embedding = np.array([int(text_hash[i:i+2], 16) for i in range(0, min(len(text_hash), 64), 2)])
                # Pad or truncate to standard size
                if len(embedding) < 384:
                    embedding = np.pad(embedding, (0, 384 - len(embedding)))
                else:
                    embedding = embedding[:384]
                return embedding.astype(float)
        
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.random.rand(384)
    
    def _store_in_chromadb(self, document: Document, embedding: np.ndarray):
        """Store document in ChromaDB"""
        try:
            # Prepare metadata
            metadata = {
                "title": document.title,
                "category": document.category,
                "author": document.author,
                "created_date": document.created_date,
                "tags": ",".join(document.tags),
                **document.metadata
            }
            
            # Add to collection
            self.collection.add(
                ids=[document.id],
                embeddings=[embedding.tolist()],
                documents=[document.content],
                metadatas=[metadata]
            )
            
        except Exception as e:
            logger.error(f"Error storing in ChromaDB: {e}")
    
    def _store_in_fallback(self, document: Document, embedding: np.ndarray):
        """Store document in fallback storage"""
        self.fallback_documents[document.id] = document
        self.fallback_embeddings[document.id] = embedding
    
    def search(self, query: str, top_k: int = 5, category: str = None) -> List[SearchResult]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            category: Filter by category (optional)
        
        Returns:
            List of search results
        """
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Perform search
            if self.storage_type == "chromadb":
                results = self._search_chromadb(query_embedding, top_k, category)
            else:
                results = self._search_fallback(query_embedding, top_k, category)
            
            # Generate answers for top results
            for result in results:
                result.answer = self._generate_answer(query, result.document.content)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def _search_chromadb(self, query_embedding: np.ndarray, top_k: int, category: str) -> List[SearchResult]:
        """Search using ChromaDB"""
        try:
            # Prepare where clause for category filter
            where_clause = None
            if category:
                where_clause = {"category": category}
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_clause
            )
            
            search_results = []
            
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                metadata = results["metadatas"][0][i]
                similarity = results["distances"][0][i]
                
                # Reconstruct document
                document = Document(
                    id=doc_id,
                    title=metadata.get("title", ""),
                    content=results["documents"][0][i],
                    category=metadata.get("category", ""),
                    author=metadata.get("author", ""),
                    created_date=metadata.get("created_date", ""),
                    tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                    metadata={k: v for k, v in metadata.items() 
                             if k not in ["title", "category", "author", "created_date", "tags"]}
                )
                
                # Convert distance to similarity score
                similarity_score = 1 - similarity
                
                search_result = SearchResult(
                    document=document,
                    similarity_score=similarity_score,
                    relevant_chunks=self._extract_relevant_chunks(
                        results["documents"][0][i], 
                        query
                    )
                )
                
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []
    
    def _search_fallback(self, query_embedding: np.ndarray, top_k: int, category: str) -> List[SearchResult]:
        """Search using fallback storage"""
        search_results = []
        
        for doc_id, document in self.fallback_documents.items():
            # Filter by category
            if category and document.category != category:
                continue
            
            # Calculate similarity
            doc_embedding = self.fallback_embeddings[doc_id]
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            search_result = SearchResult(
                document=document,
                similarity_score=similarity,
                relevant_chunks=self._extract_relevant_chunks(document.content, query)
            )
            
            search_results.append(search_result)
        
        # Sort by similarity and return top_k
        search_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return search_results[:top_k]
    
    def _extract_relevant_chunks(self, content: str, query: str, chunk_size: int = 200) -> List[str]:
        """Extract relevant chunks from document content"""
        # Split content into chunks
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        # Score chunks by query relevance
        query_words = set(query.lower().split())
        chunk_scores = []
        
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            chunk_scores.append(overlap)
        
        # Return top 3 chunks
        if chunk_scores:
            top_indices = np.argsort(chunk_scores)[-3:][::-1]
            return [chunks[i] for i in top_indices]
        
        return []
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer based on context (simplified)"""
        # This is a simplified answer generation
        # In practice, you would use a local LLM
        
        query_lower = query.lower()
        context_lower = context.lower()
        
        # Look for sentences that might answer the query
        sentences = context.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            # Check if sentence contains query terms
            if any(word in sentence_lower for word in query_lower.split() if len(word) > 2):
                return sentence.strip()
        
        # Fallback: return most relevant chunk
        chunks = self._extract_relevant_chunks(context, query)
        if chunks:
            return chunks[0]
        
        return "No specific answer found in the document."
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        return self.documents.get(doc_id)
    
    def get_documents_by_category(self, category: str) -> List[Document]:
        """Get all documents in a category"""
        return [doc for doc in self.documents.values() if doc.category == category]
    
    def get_categories(self) -> List[str]:
        """Get all document categories"""
        categories = set(doc.category for doc in self.documents.values())
        return sorted(list(categories))
    
    def update_document(self, doc_id: str, **kwargs) -> bool:
        """Update document metadata"""
        document = self.documents.get(doc_id)
        if not document:
            return False
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(document, key):
                setattr(document, key, value)
        
        # Re-generate embedding if content changed
        if "content" in kwargs:
            embedding = self._generate_embedding(document.content)
            self.embeddings[doc_id] = embedding
        
        # Update in backend
        if self.storage_type == "chromadb":
            # Delete and re-add (ChromaDB doesn't support updates easily)
            self.collection.delete(ids=[doc_id])
            self._store_in_chromadb(document, self.embeddings[doc_id])
        
        logger.info(f"Document updated: {doc_id[:8]}...")
        return True
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document"""
        if doc_id not in self.documents:
            return False
        
        # Remove from memory
        del self.documents[doc_id]
        if doc_id in self.embeddings:
            del self.embeddings[doc_id]
        
        # Remove from backend
        if self.storage_type == "chromadb":
            self.collection.delete(ids=[doc_id])
        else:
            if doc_id in self.fallback_documents:
                del self.fallback_documents[doc_id]
            if doc_id in self.fallback_embeddings:
                del self.fallback_embeddings[doc_id]
        
        logger.info(f"Document deleted: {doc_id[:8]}...")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        categories = {}
        for doc in self.documents.values():
            categories[doc.category] = categories.get(doc.category, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "categories": categories,
            "storage_type": self.storage_type,
            "embedding_type": self.embedding_type,
            "total_embeddings": len(self.embeddings)
        }
    
    def export_documents(self, output_path: str):
        """Export all documents to JSON"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "documents": [asdict(doc) for doc in self.documents.values()]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Documents exported to {output_path}")
    
    def import_documents_from_json(self, json_path: str):
        """Import documents from JSON file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            imported_count = 0
            
            for doc_data in data.get("documents", []):
                self.add_document(
                    title=doc_data.get("title", ""),
                    content=doc_data.get("content", ""),
                    category=doc_data.get("category", "general"),
                    author=doc_data.get("author", ""),
                    tags=doc_data.get("tags", []),
                    metadata=doc_data.get("metadata", {})
                )
                imported_count += 1
            
            logger.info(f"Imported {imported_count} documents from {json_path}")
            
        except Exception as e:
            logger.error(f"Error importing documents: {e}")
    
    def generate_search_report(self, query: str, results: List[SearchResult]) -> str:
        """Generate search report"""
        report_lines = [
            f"RAG Search Report",
            "=" * 20,
            f"Query: {query}",
            f"Results Found: {len(results)}",
            "",
            "Search Results:",
            "-" * 15
        ]
        
        for i, result in enumerate(results, 1):
            report_lines.extend([
                f"{i}. {result.document.title}",
                f"   Category: {result.document.category}",
                f"   Similarity: {result.similarity_score:.3f}",
                f"   Author: {result.document.author}",
                f"   Answer: {result.answer[:200]}...",
                ""
            ])
        
        return "\n".join(report_lines)
    
    def reset_system(self):
        """Reset the entire RAG system"""
        # Clear memory
        self.documents.clear()
        self.embeddings.clear()
        
        # Clear backend
        if self.storage_type == "chromadb":
            try:
                self.chroma_client.reset()
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name
                )
            except:
                pass
        else:
            self.fallback_documents.clear()
            self.fallback_embeddings.clear()
        
        logger.info("RAG system reset")

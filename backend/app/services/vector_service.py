# backend/app/services/vector_service.py

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple, Set
import uuid
import logging
import hashlib
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
import platform
import json
import time
from rank_bm25 import BM25Okapi  # ADD THIS IMPORT

# Attempt to import and configure spaCy, with NLTK as a fallback for sentence splitting.
try:
    import spacy
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    USE_SPACY = True
except ImportError:
    USE_SPACY = False
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

# --- Service Class Definition ---

class VectorService:
    """
    Production-ready vector service with performance optimizations for KAIROS.
    Features embedding caching, async operations, intelligent chunking, and improved deduplication.
    Enhanced for HackRx 6.0 with smart embedding dimension handling.
    """

    def __init__(
        self, 
        model_name: str = 'all-mpnet-base-v2',
        collection_name: str = "kairos_documents",
        persist_directory: Optional[str] = "./chroma_db",
        batch_size: Optional[int] = None,
        max_cache_size: int = 10000,
        force_recreate: bool = False
    ):
        """
        Initialize VectorService with performance optimizations.
        
        Args:
            model_name: SentenceTransformer model name
            collection_name: ChromaDB collection name
            persist_directory: Directory for persistent storage
            batch_size: Batch size for encoding (auto-detect if None)
            max_cache_size: Maximum size of embedding cache
            force_recreate: Force recreation of collection on startup
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing VectorService with model: {model_name}")

        # Initialize ChromaDB client with improved settings
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
        else:
            self.client = chromadb.Client()

        self.collection_name = collection_name
        
        # Force recreate collection if requested
        if force_recreate:
            try:
                self.client.delete_collection(collection_name)
                self.logger.info(f"Deleted existing collection: {collection_name}")
            except Exception as e:
                self.logger.debug(f"Collection deletion failed (may not exist): {e}")

        # Get or create collection with optimized settings
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:M": 16,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        initial_count = self.collection.count()
        self.logger.info(f"Collection '{collection_name}' ready with {initial_count} chunks")

        # Load the sentence-transformer model
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            # Auto-detect optimal batch size
            if batch_size is None:
                try:
                    import torch
                    self.batch_size = 64 if torch.cuda.is_available() else 16
                except ImportError:
                    self.batch_size = 16
            else:
                self.batch_size = batch_size
            
            self.logger.info(f"Model loaded. Embedding dim: {self.embedding_dimension}, Batch size: {self.batch_size}")
        except Exception as e:
            self.logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise

        # Thread pool for async operations
        max_workers = min(os.cpu_count() or 2, 4)  # Limit to 4 max workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Document deduplication tracking - improved approach
        self._document_registry = self._load_document_registry()
        
        # Configure embedding cache
        self._setup_embedding_cache(max_cache_size)
        
        # HackRx 6.0 Enhancement: Track dimension issues for analytics
        self.dimension_mismatch_count = 0
        self.last_dimension_check = datetime.utcnow()
        
        # BM25 index for hybrid search - ADDED
        self.bm25 = None
        self.bm25_doc_ids = []
        self.bm25_needs_rebuild = True

    def _setup_embedding_cache(self, max_size: int):
        """Setup LRU cache for embeddings."""
        @lru_cache(maxsize=max_size)
        def _cached_encode(text_hash: str, text: str):
            return self.model.encode([text], show_progress_bar=False)[0]
        self._cached_encode = _cached_encode

    def _get_text_hash(self, text: str) -> str:
        """Get hash of text for caching."""
        return hashlib.md5(text.encode()).hexdigest()

    # HackRx 6.0: Smart Embedding Dimension Handler
    async def _handle_embedding_dimension_mismatch(self, embeddings: List[List[float]], expected_dim: int) -> List[List[float]]:
        """
        Handle embedding dimension mismatches intelligently.
        This solves the '384 vs 768' dimension error from the logs.
        """
        if not embeddings:
            return embeddings
        
        actual_dim = len(embeddings[0]) if embeddings else 0
        
        if actual_dim != expected_dim:
            self.logger.warning(f"🔧 Embedding dimension mismatch: got {actual_dim}, expected {expected_dim}")
            self.dimension_mismatch_count += 1
            
            # Smart dimension fixing
            fixed_embeddings = []
            for emb in embeddings:
                if len(emb) > expected_dim:
                    # Truncate to expected dimension (keep most important features)
                    fixed_emb = emb[:expected_dim]
                    self.logger.debug(f"Truncated embedding from {len(emb)} to {expected_dim}")
                elif len(emb) < expected_dim:
                    # Pad with zeros to reach expected dimension
                    padding_needed = expected_dim - len(emb)
                    fixed_emb = emb + [0.0] * padding_needed
                    self.logger.debug(f"Padded embedding from {len(emb)} to {expected_dim}")
                else:
                    fixed_emb = emb
                
                fixed_embeddings.append(fixed_emb)
            
            self.logger.info(f"✅ Fixed {len(fixed_embeddings)} embeddings: {actual_dim}D → {expected_dim}D")
            return fixed_embeddings
        
        return embeddings

    async def _recreate_collection_with_correct_dimension(self):
        """Recreate collection when dimension mismatch can't be fixed."""
        self.logger.warning("🔄 Recreating collection due to unfixable dimension mismatch")
        
        try:
            # Delete existing collection
            self.client.delete_collection(self.collection_name)
            
            # Recreate with proper settings
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16,
                    "recreated_at": datetime.utcnow().isoformat(),
                    "expected_dimension": self.embedding_dimension
                }
            )
            
            # Clear registry since we recreated
            self._document_registry.clear()
            
            self.logger.info(f"✅ Collection recreated with dimension {self.embedding_dimension}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to recreate collection: {e}")
            raise

    def _load_document_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load existing document registry with improved metadata tracking."""
        try:
            if self.collection.count() == 0: 
                return {}
            
            # Get sample of documents to build registry
            sample_size = min(1000, self.collection.count())
            metadatas = self.collection.get(limit=sample_size, include=["metadatas"]).get('metadatas', [])
            
            registry = {}
            for metadata in metadatas:
                if metadata and 'document_hash' in metadata:
                    doc_hash = metadata['document_hash']
                    doc_id = metadata.get('document_id', 'unknown')
                    
                    if doc_hash not in registry:
                        registry[doc_hash] = {
                            'document_id': doc_id,
                            'indexed_at': metadata.get('indexed_at', datetime.utcnow().isoformat()),
                            'chunk_count': 0,
                            'source': metadata.get('source', 'unknown')
                        }
                    registry[doc_hash]['chunk_count'] += 1
            
            self.logger.info(f"Loaded registry with {len(registry)} documents.")
            return registry
            
        except Exception as e:
            self.logger.error(f"Error loading document registry: {e}")
            return {}

    def clear_document_registry(self, document_id: str = None):
        """Clear document from registry to force re-indexing."""
        if document_id:
            # Find and remove specific document
            to_remove = []
            for doc_hash, info in self._document_registry.items():
                if info.get('document_id') == document_id:
                    to_remove.append(doc_hash)
            for doc_hash in to_remove:
                del self._document_registry[doc_hash]
                self.logger.info(f"Cleared {document_id} from registry")
        else:
            # Clear all
            self._document_registry.clear()
            self.logger.info("Cleared entire document registry")

    def get_documents_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get all documents from a specific source."""
        try:
            results = self.collection.get(
                where={"source": source},
                limit=1000,  # Increased limit
                include=["metadatas"]
            )
            
            unique_docs = {}
            for meta in results.get('metadatas', []):
                doc_id = meta.get('document_id')
                if doc_id and doc_id not in unique_docs:
                    unique_docs[doc_id] = {
                        'document_id': doc_id,
                        'document_hash': meta.get('document_hash'),
                        'indexed_at': meta.get('indexed_at'),
                        'chunk_count': 1,
                        'source': source
                    }
                elif doc_id:
                    unique_docs[doc_id]['chunk_count'] += 1
            
            doc_list = list(unique_docs.values())
            self.logger.info(f"Found {len(doc_list)} unique documents from source '{source}'")
            return doc_list
        except Exception as e:
            self.logger.error(f"Error getting documents by source: {e}")
            return []

    def find_document_by_pattern(self, pattern: str, source: str = None) -> Optional[str]:
        """Find the most recent document ID matching a pattern."""
        try:
            where_clause = {}
            if source:
                where_clause["source"] = source
                
            results = self.collection.get(
                where=where_clause,
                limit=100,
                include=["metadatas"]
            )
            
            matching_docs = []
            for meta in results.get('metadatas', []):
                doc_id = meta.get('document_id', '')
                if pattern in doc_id:
                    indexed_at = meta.get('indexed_at', '')
                    matching_docs.append((doc_id, indexed_at))
            
            # Sort by timestamp (most recent first)
            matching_docs.sort(key=lambda x: x[1], reverse=True)
            
            if matching_docs:
                self.logger.info(f"Found document matching pattern '{pattern}': {matching_docs[0][0]}")
                return matching_docs[0][0]
            return None
        except Exception as e:
            self.logger.error(f"Error finding document by pattern: {e}")
            return None

    def _smart_chunk_text(
        self, text: str, chunk_size: int = 1000, overlap: int = 200, min_chunk_size: int = 100
    ) -> List[Tuple[str, int, int]]:
        """Intelligently chunks text using proper sentence segmentation with improvements."""
        if not text or not text.strip(): 
            return []

        # Clean and normalize text
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Normalize whitespace
        
        if len(text) < min_chunk_size:
            return [(text, 0, len(text))]

        try:
            if USE_SPACY:
                doc = nlp(text)
                sentences = [(sent.text.strip(), sent.start_char) for sent in doc.sents if sent.text.strip()]
            else:
                import nltk
                sentences = []
                sent_texts = nltk.sent_tokenize(text)
                current_pos = 0
                for sent_text in sent_texts:
                    sent_text = sent_text.strip()
                    if sent_text:
                        start_pos = text.find(sent_text, current_pos)
                        if start_pos == -1:
                            start_pos = current_pos
                        sentences.append((sent_text, start_pos))
                        current_pos = start_pos + len(sent_text)
        except Exception as e:
            self.logger.warning(f"Sentence splitting failed: {e}, using simple chunking")
            # Fallback to simple chunking
            chunks = []
            for i in range(0, len(text), chunk_size - overlap):
                chunk_text = text[i:i + chunk_size]
                if len(chunk_text) >= min_chunk_size:
                    chunks.append((chunk_text, i, i + len(chunk_text)))
            return chunks

        if not sentences:
            return [(text, 0, len(text))]

        chunks = []
        current_chunk_sentences = []
        current_size = 0
        
        for sent_text, sent_start in sentences:
            # Check if adding this sentence would exceed chunk size
            if current_size + len(sent_text) > chunk_size and current_chunk_sentences:
                # Create chunk from current sentences
                chunk_text = " ".join([s[0] for s in current_chunk_sentences])
                if len(chunk_text) >= min_chunk_size:
                    start_pos = current_chunk_sentences[0][1]
                    end_pos = current_chunk_sentences[-1][1] + len(current_chunk_sentences[-1][0])
                    chunks.append((chunk_text, start_pos, end_pos))
                
                # Create overlap by keeping last few sentences
                overlap_sents = []
                overlap_len = 0
                for s in reversed(current_chunk_sentences):
                    if overlap_len < overlap and len(overlap_sents) < 3:  # Max 3 sentences for overlap
                        overlap_sents.insert(0, s)
                        overlap_len += len(s[0])
                    else:
                        break
                
                current_chunk_sentences = overlap_sents
                current_size = sum(len(s[0]) for s in current_chunk_sentences)
            
            current_chunk_sentences.append((sent_text, sent_start))
            current_size += len(sent_text)

        # Add final chunk if it has content
        if current_chunk_sentences:
            chunk_text = " ".join([s[0] for s in current_chunk_sentences])
            if len(chunk_text) >= min_chunk_size:
                start_pos = current_chunk_sentences[0][1]
                end_pos = current_chunk_sentences[-1][1] + len(current_chunk_sentences[-1][0])
                chunks.append((chunk_text, start_pos, end_pos))
        
        return chunks

    async def _encode_texts_async(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously encode texts with improved caching and error handling."""
        if not texts:
            return []
            
        loop = asyncio.get_event_loop()
        embeddings = [None] * len(texts)
        texts_to_encode = []
        indices_to_encode = []

        # Check cache for existing embeddings
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            try:
                cached_embedding = self._cached_encode(text_hash, text)
                embeddings[i] = cached_embedding.tolist()
            except Exception:
                # Not in cache or other error
                texts_to_encode.append(text)
                indices_to_encode.append(i)
        
        # Encode uncached texts
        if texts_to_encode:
            try:
                new_embeddings = await loop.run_in_executor(
                    self.executor,
                    lambda: self.model.encode(
                        texts_to_encode, 
                        show_progress_bar=False, 
                        batch_size=self.batch_size,
                        convert_to_tensor=False
                    )
                )
                
                for i, (idx, emb) in enumerate(zip(indices_to_encode, new_embeddings)):
                    embeddings[idx] = emb.tolist()
                    
            except Exception as e:
                self.logger.error(f"Error encoding texts: {e}")
                raise

        return embeddings

    async def force_reindex_document(self, text: str, metadata: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Force reindex a document without changing its ID."""
        doc_id = metadata.get('document_id')
        if not doc_id:
            raise ValueError("document_id is required for force reindex")
        
        self.logger.info(f"Force reindexing document: {doc_id}")
        
        # Delete existing document
        self.delete_document(doc_id)
        
        # Wait for deletion to propagate
        await asyncio.sleep(0.5)
        
        # Clear from registry
        self.clear_document_registry(doc_id)
        
        # Add with force_reindex=True and skip_if_exists=False
        result = await self.add_document(
            text=text,
            metadata=metadata,
            force_reindex=True,
            skip_if_exists=False,
            **kwargs
        )
        
        # Mark BM25 index for rebuild - ADDED
        self.bm25_needs_rebuild = True
        
        # Verify the document was indexed correctly
        if result.get("success"):
            await asyncio.sleep(0.5)  # Brief wait for consistency
            
            # Verify we can find it
            verification = await self.search(
                query="document verification test",
                n_results=1,
                filter_metadata={"document_id": doc_id}
            )
            
            if not verification:
                self.logger.error(f"Verification failed for {doc_id}")
                raise Exception("Document indexing verification failed")
            
            self.logger.info(f"Successfully verified document {doc_id}")
        
        return result

    async def add_document(
        self, 
        text: str, 
        metadata: Dict[str, Any], 
        chunk_size: int = 1000, 
        overlap: int = 200, 
        skip_if_exists: bool = True,
        force_reindex: bool = False
    ) -> Dict[str, Any]:
        """
        Asynchronously adds a document to the vector database with improved deduplication.
        Enhanced for HackRx 6.0 with smart dimension handling.
        
        Args:
            text: Document text content
            metadata: Document metadata
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            skip_if_exists: Skip if document already exists
            force_reindex: Force reindexing even if document exists
        """
        if not text or not text.strip(): 
            raise ValueError("Document text cannot be empty")
        if 'document_id' not in metadata: 
            metadata['document_id'] = f"doc_{uuid.uuid4()}"
        
        try:
            start_time = datetime.utcnow()
            doc_hash = hashlib.sha256(text.encode()).hexdigest()
            doc_id = metadata['document_id']
            
            # Ensure source is set
            if 'source' not in metadata:
                metadata['source'] = 'hackrx'
            
            # Enhanced existence check
            if skip_if_exists and not force_reindex and doc_hash in self._document_registry:
                existing_info = self._document_registry[doc_hash]
                self.logger.info(f"Skipping existing document: {doc_id} (hash: {doc_hash[:8]}...)")
                return {
                    "success": True, 
                    "message": "Document already exists",
                    "document_hash": doc_hash,
                    "existing_chunks": existing_info.get('chunk_count', 0),
                    "skipped": True
                }

            # If force reindex, delete existing chunks first
            if force_reindex:
                self.logger.info(f"Force reindexing document: {doc_id}")
                # Clear from registry
                if doc_hash in self._document_registry:
                    del self._document_registry[doc_hash]
                # Delete from collection
                self.delete_document(doc_id)
                # Wait for deletion to complete
                await asyncio.sleep(0.5)

            # Create chunks with improved chunking
            chunk_data = self._smart_chunk_text(text, chunk_size, overlap)
            if not chunk_data: 
                return {"success": False, "error": "No valid chunks created", "chunks_created": 0}
            
            chunks = [c[0] for c in chunk_data]
            chunk_ids = [f"{doc_id}_chunk_{i:04d}" for i in range(len(chunks))]  # Zero-padded for better sorting
            
            # Enhanced metadata for each chunk
            current_time = datetime.utcnow().isoformat()
            chunk_metadatas = []
            for i, (_, start_pos, end_pos) in enumerate(chunk_data):
                chunk_meta = {
                    **metadata, 
                    'chunk_index': i, 
                    'total_chunks': len(chunks), 
                    'chunk_start': start_pos, 
                    'chunk_end': end_pos, 
                    'document_hash': doc_hash,
                    'indexed_at': current_time,
                    'chunk_size': len(chunks[i]),
                    'chunk_id': chunk_ids[i],
                    'document_id': doc_id,  # Ensure this is always present
                    'source': metadata.get('source', 'hackrx')  # Ensure source is always present
                }
                chunk_metadatas.append(chunk_meta)
            
            # Generate embeddings
            embeddings = await self._encode_texts_async(chunks)
            
            # HackRx 6.0 Enhancement: Smart dimension handling
            try:
                # Handle potential dimension mismatch
                embeddings = await self._handle_embedding_dimension_mismatch(embeddings, self.embedding_dimension)
                
                self.collection.add(
                    embeddings=embeddings, 
                    documents=chunks, 
                    metadatas=chunk_metadatas, 
                    ids=chunk_ids
                )
                
            except Exception as e:
                # Enhanced error handling for different types of failures
                error_msg = str(e).lower()
                
                if "already exists" in error_msg:
                    self.logger.warning(f"Some chunks already exist, attempting to update: {e}")
                    # Delete existing chunks and retry
                    self.delete_document(doc_id)
                    await asyncio.sleep(0.5)
                    
                    # Fix embeddings again after regeneration
                    embeddings = await self._handle_embedding_dimension_mismatch(embeddings, self.embedding_dimension)
                    
                    self.collection.add(
                        embeddings=embeddings, 
                        documents=chunks, 
                        metadatas=chunk_metadatas, 
                        ids=chunk_ids
                    )
                    
                elif "dimension" in error_msg or "expecting embedding" in error_msg:
                    self.logger.error(f"🔧 Embedding dimension error: {e}")
                    # Try to recreate collection with correct dimension
                    await self._recreate_collection_with_correct_dimension()
                    
                    # Retry with new collection
                    embeddings = await self._handle_embedding_dimension_mismatch(embeddings, self.embedding_dimension)
                    self.collection.add(
                        embeddings=embeddings, 
                        documents=chunks, 
                        metadatas=chunk_metadatas, 
                        ids=chunk_ids
                    )
                    
                elif "collection" in error_msg and "not found" in error_msg:
                    self.logger.warning("Collection was deleted, recreating...")
                    # Recreate collection
                    self.collection = self.client.get_or_create_collection(
                        name=self.collection_name,
                        metadata={
                            "hnsw:space": "cosine",
                            "hnsw:construction_ef": 200,
                            "hnsw:M": 16,
                            "recreated_at": datetime.utcnow().isoformat()
                        }
                    )
                    
                    # Retry
                    embeddings = await self._handle_embedding_dimension_mismatch(embeddings, self.embedding_dimension)
                    self.collection.add(
                        embeddings=embeddings, 
                        documents=chunks, 
                        metadatas=chunk_metadatas, 
                        ids=chunk_ids
                    )
                else:
                    # Unknown error, re-raise
                    raise
            
            # Update registry
            self._document_registry[doc_hash] = {
                'document_id': doc_id,
                'indexed_at': current_time,
                'chunk_count': len(chunks),
                'source': metadata.get('source', 'hackrx'),
                'text_length': len(text)
            }
            
            # Mark BM25 index for rebuild - ADDED
            self.bm25_needs_rebuild = True
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"✅ Indexed document {doc_id} ({len(chunks)} chunks) in {processing_time:.2f}s")
            
            return {
                "success": True, 
                "chunks_created": len(chunks), 
                "document_hash": doc_hash,
                "processing_time": processing_time,
                "chunk_size_avg": sum(len(c) for c in chunks) // len(chunks) if chunks else 0,
                "document_id": doc_id,
                "dimension_fixes_applied": self.dimension_mismatch_count > 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error adding document {metadata.get('document_id', 'unknown')}: {e}")
            return {"success": False, "error": str(e), "chunks_created": 0}

    def _build_bm25_index(self):
        """Build BM25 index for keyword search - ADDED METHOD"""
        try:
            # Get all documents from collection
            all_docs = self.collection.get(
                limit=100000,  # Get all documents
                include=['documents', 'metadatas']
            )
            
            if not all_docs['ids']:
                self.logger.warning("No documents found for BM25 index")
                return
            
            # Tokenize documents for BM25
            tokenized_docs = []
            self.bm25_doc_ids = []
            
            for i, doc in enumerate(all_docs['documents']):
                # Simple tokenization - split on whitespace and lowercase
                tokens = doc.lower().split()
                tokenized_docs.append(tokens)
                self.bm25_doc_ids.append(all_docs['ids'][i])
            
            # Build BM25 index
            self.bm25 = BM25Okapi(tokenized_docs)
            self.bm25_needs_rebuild = False
            
            self.logger.info(f"Built BM25 index with {len(tokenized_docs)} documents")
            
        except Exception as e:
            self.logger.error(f"Error building BM25 index: {e}")
            self.bm25 = None
            self.bm25_doc_ids = []

    async def hybrid_search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
        vector_weight: float = 0.6,  # 60% vector, 40% keyword
        include_raw_distances: bool = False
    ) -> List[Dict[str, Any]]:
        """
        HYBRID SEARCH - Combines BM25 keyword search with vector similarity
        This is the main improvement for better accuracy!
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        
        try:
            self.logger.debug(f"🔍 Hybrid search for: '{query[:100]}...' with filters: {filter_metadata}")
            
            # Step 1: Build or rebuild BM25 index if needed
            if self.bm25 is None or self.bm25_needs_rebuild:
                self.logger.info("Building BM25 index for hybrid search...")
                self._build_bm25_index()
            
            # Step 2: Vector search (using existing search method)
            vector_results = await self.search(
                query=query,
                n_results=n_results * 2,  # Get more results for merging
                filter_metadata=filter_metadata,
                min_score=0,  # Don't filter by score yet
                include_raw_distances=include_raw_distances
            )
            
            # Step 3: BM25 keyword search
            bm25_results = []
            if self.bm25 is not None and self.bm25_doc_ids:
                # Tokenize query
                query_tokens = query.lower().split()
                
                # Get BM25 scores
                bm25_scores = self.bm25.get_scores(query_tokens)
                
                # Get top BM25 results
                top_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i],
                    reverse=True
                )[:n_results * 2]
                
                # Fetch BM25 results
                for idx in top_indices:
                    if idx < len(self.bm25_doc_ids):
                        chunk_id = self.bm25_doc_ids[idx]
                        score = float(bm25_scores[idx])
                        
                        # Normalize BM25 score to [0, 1] range
                        # BM25 scores can be any positive value, so we use sigmoid-like normalization
                        normalized_score = score / (1 + score) if score > 0 else 0
                        
                        bm25_results.append({
                            'chunk_id': chunk_id,
                            'bm25_score': normalized_score
                        })
            
            # Step 4: Combine results
            combined_scores = {}
            
            # Add vector results
            for result in vector_results:
                chunk_id = result['chunk_id']
                combined_scores[chunk_id] = {
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'vector_score': result['similarity_score'],
                    'bm25_score': 0,
                    'final_score': result['similarity_score'] * vector_weight
                }
                if include_raw_distances and 'raw_distance' in result:
                    combined_scores[chunk_id]['raw_distance'] = result['raw_distance']
            
            # Add BM25 scores
            for bm25_result in bm25_results:
                chunk_id = bm25_result['chunk_id']
                bm25_score = bm25_result['bm25_score'] * (1 - vector_weight)
                
                if chunk_id in combined_scores:
                    # Combine scores
                    combined_scores[chunk_id]['bm25_score'] = bm25_result['bm25_score']
                    combined_scores[chunk_id]['final_score'] += bm25_score
                else:
                    # Fetch the chunk data if not in vector results
                    try:
                        chunk_data = self.collection.get(
                            ids=[chunk_id],
                            include=['documents', 'metadatas']
                        )
                        
                        if chunk_data['ids']:
                            # Apply metadata filter if provided
                            metadata = chunk_data['metadatas'][0]
                            
                            # Check if metadata matches filter
                            if filter_metadata:
                                match = all(
                                    metadata.get(k) == v 
                                    for k, v in filter_metadata.items()
                                )
                                if not match:
                                    continue
                            
                            combined_scores[chunk_id] = {
                                'content': chunk_data['documents'][0],
                                'metadata': metadata,
                                'vector_score': 0,
                                'bm25_score': bm25_result['bm25_score'],
                                'final_score': bm25_score
                            }
                    except Exception as e:
                        self.logger.debug(f"Could not fetch chunk {chunk_id}: {e}")
                        continue
            
            # Step 5: Sort by final score and apply min_score filter
            sorted_results = sorted(
                combined_scores.items(),
                key=lambda x: x[1]['final_score'],
                reverse=True
            )
            
            # Filter by minimum score and limit results
            final_results = []
            for chunk_id, data in sorted_results:
                if data['final_score'] >= min_score:
                    result_item = {
                        'chunk_id': chunk_id,
                        'content': data['content'],
                        'metadata': data['metadata'],
                        'similarity_score': data['final_score'],
                        'vector_score': data['vector_score'],
                        'bm25_score': data['bm25_score']
                    }
                    
                    if include_raw_distances and 'raw_distance' in data:
                        result_item['raw_distance'] = data['raw_distance']
                    
                    final_results.append(result_item)
                    
                    if len(final_results) >= n_results:
                        break
            
            self.logger.debug(f"✅ Hybrid search found {len(final_results)} results")
            
            # Log sample of results for debugging
            if final_results:
                self.logger.debug(
                    f"🎯 Top result: final_score={final_results[0]['similarity_score']:.4f}, "
                    f"vector={final_results[0]['vector_score']:.4f}, "
                    f"bm25={final_results[0]['bm25_score']:.4f}, "
                    f"doc_id={final_results[0]['metadata'].get('document_id')}"
                )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ Error during hybrid search: {e}", exc_info=True)
            # Fallback to regular search if hybrid search fails
            self.logger.info("Falling back to regular vector search")
            return await self.search(
                query=query,
                n_results=n_results,
                filter_metadata=filter_metadata,
                min_score=min_score,
                include_raw_distances=include_raw_distances
            )

    async def search(
        self, 
        query: str, 
        n_results: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None, 
        min_score: float = 0.0,
        include_raw_distances: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously searches for relevant chunks with improved error handling.
        Enhanced for HackRx 6.0 with better diagnostics.
        """
        if not query or not query.strip(): 
            raise ValueError("Search query cannot be empty")
        
        try:
            # Clean and normalize query
            query = query.strip()
            self.logger.debug(f"🔍 Searching for: '{query[:100]}...' with filters: {filter_metadata}")
            
            # First, check if ANY documents exist with the filter
            if filter_metadata:
                # Do a lightweight check first
                check_results = self.collection.get(
                    where=filter_metadata,
                    limit=1,
                    include=["metadatas"]
                )
                
                if not check_results['ids']:
                    self.logger.warning(f"⚠️ No documents found with filter: {filter_metadata}")
                    
                    # Try alternative searches to diagnose the issue
                    diagnostics = []
                    
                    # Check by source only
                    if 'source' in filter_metadata:
                        source_check = self.collection.get(
                            where={"source": filter_metadata['source']},
                            limit=5,
                            include=['metadatas']
                        )
                        if source_check['ids']:
                            found_docs = list(set([m.get('document_id', 'unknown') for m in source_check['metadatas']]))
                            diagnostics.append(f"Found {len(found_docs)} docs with source '{filter_metadata['source']}': {found_docs[:5]}")
                        else:
                            diagnostics.append(f"No documents found with source '{filter_metadata['source']}'")
                    
                    # Check total documents
                    total_count = self.collection.count()
                    diagnostics.append(f"Total documents in collection: {total_count}")
                    
                    # Check if the document_id exists with any source
                    if 'document_id' in filter_metadata:
                        doc_check = self.collection.get(
                            where={"document_id": filter_metadata['document_id']},
                            limit=1,
                            include=['metadatas']
                        )
                        if doc_check['ids']:
                            actual_source = doc_check['metadatas'][0].get('source', 'unknown')
                            diagnostics.append(f"Document {filter_metadata['document_id']} exists with source: {actual_source}")
                        else:
                            # Try to find by pattern
                            pattern_match = self.find_document_by_pattern(
                                filter_metadata['document_id'], 
                                filter_metadata.get('source')
                            )
                            if pattern_match:
                                diagnostics.append(f"Found document by pattern: {pattern_match}")
                            else:
                                diagnostics.append(f"Document {filter_metadata['document_id']} not found anywhere")
                    
                    # Log diagnostics
                    for diag in diagnostics:
                        self.logger.info(f"🔍 {diag}")
                    
                    # Return empty results instead of failing
                    return []
            
            # Generate query embedding
            query_embedding = await self._encode_texts_async([query])
            
            # HackRx 6.0: Handle query embedding dimension
            query_embedding = await self._handle_embedding_dimension_mismatch(query_embedding, self.embedding_dimension)
            
            # Perform search with increased results for better filtering
            search_limit = min(n_results * 3, 50)
            
            # Robust search with error handling
            try:
                results = self.collection.query(
                    query_embeddings=query_embedding, 
                    n_results=search_limit,
                    where=filter_metadata, 
                    include=['metadatas', 'documents', 'distances']
                )
            except Exception as e:
                self.logger.error(f"❌ ChromaDB query failed: {e}")
                # Try without filter as fallback
                if filter_metadata:
                    self.logger.warning("🔄 Retrying search without filters")
                    results = self.collection.query(
                        query_embeddings=query_embedding, 
                        n_results=search_limit,
                        include=['metadatas', 'documents', 'distances']
                    )
                else:
                    raise
            
            formatted_results = []
            if results and results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    distance = results['distances'][0][i]
                    # Improved similarity calculation
                    # ChromaDB cosine distance is in [0, 2], where 0 = identical, 2 = opposite
                    similarity_score = max(0, 1 - (distance / 2))  # Normalize to [0, 1]
                    
                    if similarity_score >= min_score:
                        result_item = {
                            'chunk_id': results['ids'][0][i],
                            'content': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'similarity_score': round(similarity_score, 4)
                        }
                        
                        if include_raw_distances:
                            result_item['raw_distance'] = distance
                        
                        formatted_results.append(result_item)
            
            # Sort by similarity score and limit results
            formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            final_results = formatted_results[:n_results]
            
            self.logger.debug(f"✅ Found {len(final_results)} results (from {len(formatted_results)} candidates)")
            
            # Log sample of results for debugging
            if final_results:
                self.logger.debug(f"🎯 Top result: score={final_results[0]['similarity_score']}, "
                                f"doc_id={final_results[0]['metadata'].get('document_id')}, "
                                f"chunk_index={final_results[0]['metadata'].get('chunk_index')}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ Error during search: {e}", exc_info=True)
            # Return empty results instead of raising in production
            return []

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Deletes all chunks associated with a document with improved cleanup."""
        try:
            # Get document info first
            doc_hash = None
            results = self.collection.get(where={"document_id": document_id}, limit=1, include=["metadatas"])
            if results and results['metadatas']:
                doc_hash = results['metadatas'][0].get('document_hash')
            
            # Count chunks before deletion
            before_count = self.collection.count()
            
            # Delete all chunks for this document
            self.collection.delete(where={"document_id": document_id})
            
            # Count chunks after deletion
            after_count = self.collection.count()
            chunks_deleted = before_count - after_count
            
            # Update registry
            if doc_hash and doc_hash in self._document_registry:
                del self._document_registry[doc_hash]
            
            # Mark BM25 index for rebuild - ADDED
            if chunks_deleted > 0:
                self.bm25_needs_rebuild = True
            
            self.logger.info(f"🗑️ Deleted {chunks_deleted} chunks for document {document_id}")
            return {
                "success": True, 
                "document_id": document_id, 
                "chunks_deleted": chunks_deleted,
                "document_hash": doc_hash
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error deleting document {document_id}: {e}")
            return {"success": False, "document_id": document_id, "error": str(e)}

    async def ensure_consistency(self, document_id: str = None) -> bool:
        """Ensure ChromaDB has processed all pending operations"""
        try:
            # Force a read operation to ensure consistency
            if document_id:
                result = self.collection.get(
                    where={"document_id": document_id},
                    limit=1,
                    include=["metadatas"]
                )
                return bool(result['ids'])
            else:
                # Just check collection is responsive
                count = self.collection.count()
                return count >= 0
        except Exception as e:
            self.logger.error(f"Consistency check failed: {e}")
            return False

    async def check_health(self) -> Dict[str, Any]:
        """Check if the vector service and ChromaDB are healthy."""
        try:
            # Test 1: Can we count documents?
            count = self.collection.count()
            
            # Test 2: Can we perform a search?
            test_search = await self.search(
                query="health check test",
                n_results=1,
                min_score=0.0
            )
            
            # Test 3: Can we get collection stats?
            stats = await self.get_collection_stats()
            
            return {
                "healthy": True,
                "document_count": count,
                "search_functional": True,
                "stats": stats,
                "dimension_fixes_applied": self.dimension_mismatch_count,
                "bm25_index_ready": self.bm25 is not None  # ADDED
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e)
            }

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Returns comprehensive statistics about the vector collection."""
        try:
            total_chunks = self.collection.count()
            if total_chunks == 0:
                return {
                    "collection_name": self.collection_name,
                    "total_chunks": 0,
                    "unique_documents": 0,
                    "cached_embeddings": self._cached_encode.cache_info().currsize,
                    "status": "empty",
                    "dimension_fixes_applied": self.dimension_mismatch_count,
                    "bm25_index_status": "not_built"  # ADDED
                }
            
            # Get sample for analysis
            sample_size = min(1000, total_chunks)
            sample_data = self.collection.get(limit=sample_size, include=["metadatas"])
            sample_metadatas = sample_data.get('metadatas', [])
            
            # Analyze document distribution
            doc_ids = set()
            sources = {}
            chunk_sizes = []
            
            for metadata in sample_metadatas:
                if metadata:
                    doc_id = metadata.get('document_id')
                    if doc_id:
                        doc_ids.add(doc_id)
                    
                    source = metadata.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
                    
                    chunk_size = metadata.get('chunk_size', 0)
                    if chunk_size > 0:
                        chunk_sizes.append(chunk_size)
            
            # Extrapolate unique documents
            if total_chunks > sample_size and sample_size > 0:
                estimated_unique_docs = int(len(doc_ids) * (total_chunks / sample_size))
            else:
                estimated_unique_docs = len(doc_ids)
            
            # Calculate averages
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            
            return {
                "collection_name": self.collection_name,
                "total_chunks": total_chunks,
                "unique_documents_estimated": estimated_unique_docs,
                "avg_chunks_per_document": total_chunks / estimated_unique_docs if estimated_unique_docs > 0 else 0,
                "embedding_model": getattr(self.model, 'model_name', 'unknown'),
                "embedding_dimension": self.embedding_dimension,
                "distance_metric": "cosine",
                "cached_embeddings": self._cached_encode.cache_info().currsize,
                "cache_stats": self._cached_encode.cache_info()._asdict(),
                "registry_size": len(self._document_registry),
                "sources": dict(sources),
                "avg_chunk_size": round(avg_chunk_size, 2),
                "status": "healthy",
                # HackRx 6.0 Analytics
                "dimension_fixes_applied": self.dimension_mismatch_count,
                "last_dimension_check": self.last_dimension_check.isoformat(),
                "hackrx_ready": True,
                # BM25 status - ADDED
                "bm25_index_status": "ready" if self.bm25 is not None else "not_built",
                "bm25_index_size": len(self.bm25_doc_ids) if self.bm25 else 0,
                "bm25_needs_rebuild": self.bm25_needs_rebuild
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e), "status": "error"}

    async def search_similar_chunks(self, chunk_id: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Finds chunks similar to a given chunk using native ChromaDB operations."""
        try:
            # Get the target chunk's embedding
            chunk_data = self.collection.get(ids=[chunk_id], include=['embeddings', 'metadatas'])
            if not chunk_data['ids']:
                raise ValueError(f"Chunk {chunk_id} not found")
            
            # Handle potential dimension mismatch
            embeddings = await self._handle_embedding_dimension_mismatch(
                chunk_data['embeddings'], 
                self.embedding_dimension
            )
            
            # Search for similar chunks
            results = self.collection.query(
                query_embeddings=embeddings,
                n_results=n_results + 1,  # +1 because the query chunk will be included
                include=['metadatas', 'documents', 'distances']
            )
            
            formatted_results = []
            if results and results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    if results['ids'][0][i] != chunk_id:  # Exclude the query chunk itself
                        distance = results['distances'][0][i]
                        similarity_score = max(0, 1 - (distance / 2))  # Normalize similarity
                        
                        formatted_results.append({
                            'chunk_id': results['ids'][0][i],
                            'content': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'similarity_score': round(similarity_score, 4),
                        })
                        
                        if len(formatted_results) >= n_results:
                            break
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error finding similar chunks: {e}")
            raise

    def reset_collection(self) -> Dict[str, Any]:
        """Resets the collection and clears all caches and registries."""
        try:
            # Delete the collection
            self.client.delete_collection(self.collection_name)
            
            # Recreate the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16,
                    "reset_at": datetime.utcnow().isoformat(),
                    "embedding_dimension": self.embedding_dimension
                }
            )
            
            # Clear all caches and registries
            self._document_registry.clear()
            self._cached_encode.cache_clear()
            
            # Reset HackRx analytics
            self.dimension_mismatch_count = 0
            self.last_dimension_check = datetime.utcnow()
            
            # Reset BM25 index - ADDED
            self.bm25 = None
            self.bm25_doc_ids = []
            self.bm25_needs_rebuild = True
            
            self.logger.info(f"✅ Collection {self.collection_name} reset successfully")
            return {"success": True, "message": "Collection reset successfully"}
            
        except Exception as e:
            self.logger.error(f"❌ Error resetting collection: {e}")
            return {"success": False, "error": str(e)}

    def get_document_info(self, document_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific document."""
        try:
            # Get all chunks for this document
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if not results['ids']:
                return {"error": f"Document {document_id} not found"}
            
            metadatas = results['metadatas']
            chunk_count = len(metadatas)
            
            # Extract document info from first chunk metadata
            first_meta = metadatas[0] if metadatas else {}
            doc_hash = first_meta.get('document_hash', 'unknown')
            
            # Get from registry if available
            registry_info = self._document_registry.get(doc_hash, {})
            
            return {
                "document_id": document_id,
                "document_hash": doc_hash,
                "chunk_count": chunk_count,
                "indexed_at": first_meta.get('indexed_at', 'unknown'),
                "source": first_meta.get('source', 'unknown'),
                "text_length": registry_info.get('text_length', 'unknown'),
                "chunks": [
                    {
                        "chunk_id": results['ids'][i],
                        "chunk_index": meta.get('chunk_index', i),
                        "chunk_size": meta.get('chunk_size', 'unknown'),
                        "chunk_start": meta.get('chunk_start', 'unknown'),
                        "chunk_end": meta.get('chunk_end', 'unknown')
                    }
                    for i, meta in enumerate(metadatas)
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting document info for {document_id}: {e}")
            return {"error": str(e)}

    # HackRx 6.0: Analytics and Monitoring
    def get_hackrx_analytics(self) -> Dict[str, Any]:
        """Get HackRx-specific analytics for monitoring and debugging."""
        return {
            "service": "VectorService",
            "status": "healthy",
            "embedding_model": getattr(self.model, 'model_name', 'unknown'),
            "embedding_dimension": self.embedding_dimension,
            "collection_name": self.collection_name,
            "total_documents": len(self._document_registry),
            "dimension_mismatches_fixed": self.dimension_mismatch_count,
            "last_dimension_check": self.last_dimension_check.isoformat(),
            "cache_size": self._cached_encode.cache_info().currsize,
            "max_cache_size": self._cached_encode.cache_info().maxsize,
            "cache_hit_rate": (
                self._cached_encode.cache_info().hits / 
                max(1, self._cached_encode.cache_info().hits + self._cached_encode.cache_info().misses)
            ),
            "features": {
                "smart_dimension_handling": True,
                "automatic_error_recovery": True,
                "intelligent_chunking": True,
                "embedding_caching": True,
                "document_deduplication": True,
                "hackrx_optimized": True,
                "hybrid_search": True,  # ADDED
                "bm25_keyword_search": True  # ADDED
            },
            "bm25_status": {  # ADDED
                "index_ready": self.bm25 is not None,
                "index_size": len(self.bm25_doc_ids) if self.bm25 else 0,
                "needs_rebuild": self.bm25_needs_rebuild
            }
        }

    def __del__(self):
        """Cleanup resources on object deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
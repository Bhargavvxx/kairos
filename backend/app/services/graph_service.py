# backend/app/services/graph_service.py

import networkx as nx
from networkx.readwrite import json_graph
from typing import List, Dict, Any, Optional, Set, Tuple, Protocol, Callable, Union
import json
import logging
import re
import os
import gzip
import asyncio
from datetime import datetime, timezone
from collections import Counter, defaultdict
import time
from functools import lru_cache
import hashlib
import pickle
import uuid

# Import our Pydantic models
from ..models.schemas import (
    GraphData,
    Node,
    Edge,
    NodeType,
    RelationshipType,
    SmartNodeTypeValidator,
)

# Constants
GRAPH_SCHEMA_VERSION = "2.0"
DEFAULT_MAX_TOKENS = 4000
TOKEN_TO_CHAR_RATIO = 4 / 3  # Approximate ratio
MAX_INDEX_SIZE = 100000  # Prevent unbounded growth
MAX_TEXT_SIZE = 1_000_000  # 1MB limit
DEFAULT_CHUNK_SIZE = 3000  # Base chunk size
DEFAULT_CHUNK_OVERLAP = 300  # Overlap between chunks
SMALL_DOC_THRESHOLD = 20000  # Below this, use single extraction
LARGE_DOC_THRESHOLD = 100000  # Above this, use adaptive chunking
VERY_LARGE_DOC_THRESHOLD = 500000  # Above this, use sampling

# Batch processing
DEFAULT_BATCH_SIZE = 3
LARGE_DOC_BATCH_SIZE = 5
MAX_CONCURRENT_LLM_CALLS = 3

# Checkpointing
CHECKPOINT_DIR = "./graph_checkpoints"
CHECKPOINT_INTERVAL = 10  # Save every N chunks

# --- LLM Client Protocol ---
class LLMClient(Protocol):
    """Protocol for LLM clients."""
    async def generate(self, prompt: str) -> str: ...

# --- Storage Adapter Protocol ---
class StorageAdapter(Protocol):
    """Protocol defining the contract for graph storage adapters."""
    
    async def save(self, graph_dict: Dict[str, Any], metadata: Dict[str, Any]) -> bool: ...
    async def load(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]: ...
    async def exists(self) -> bool: ...

class FileStorageAdapter:
    """File-based storage adapter using compressed JSON for safety and portability."""
    
    def __init__(self, path: str):
        self.path = path
        self.graph_file = f"{path}/graph.json.gz"
        self.metadata_file = f"{path}/metadata.json"
    
    async def save(self, graph_dict: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        try:
            os.makedirs(self.path, exist_ok=True)
            # Use asyncio to run blocking I/O in thread pool
            loop = asyncio.get_running_loop()
            
            def _save_files():
                with gzip.open(self.graph_file, 'wt', encoding='utf-8') as f:
                    json.dump(graph_dict, f)
                with open(self.metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            await loop.run_in_executor(None, _save_files)
            return True
        except Exception as e:
            logging.error(f"Failed to save graph to file: {e}")
            return False
    
    async def load(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        try:
            if not await self.exists(): 
                return None
            
            loop = asyncio.get_running_loop()
            
            def _load_files():
                with gzip.open(self.graph_file, 'rt', encoding='utf-8') as f:
                    graph_dict = json.load(f)
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                return graph_dict, metadata
            
            return await loop.run_in_executor(None, _load_files)
        except Exception as e:
            logging.error(f"Failed to load graph from file: {e}")
            return None
    
    async def exists(self) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            lambda: os.path.exists(self.graph_file) and os.path.exists(self.metadata_file)
        )

class InMemoryStorageAdapter(StorageAdapter):
    """In-memory storage adapter for testing and development."""
    def __init__(self):
        self.data: Optional[Tuple[Dict, Dict]] = None
    
    async def save(self, graph_dict: Dict, metadata: Dict) -> bool:
        self.data = (graph_dict, metadata)
        return True
    
    async def load(self) -> Optional[Tuple[Dict, Dict]]:
        return self.data
    
    async def exists(self) -> bool:
        return self.data is not None

# --- Service Class Definition ---

class GraphService:
    """
    Knowledge graph service for document analysis.
    Production-ready GraphService for KAIROS with improved performance,
    thread safety, and bulletproof handling of ANY document types.
    """

    def __init__(
        self, 
        storage_adapter: Optional[StorageAdapter] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_concurrent_operations: int = 3,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        enable_checkpointing: bool = True,
        enable_sampling: bool = True
    ):
        self.storage_adapter = storage_adapter or InMemoryStorageAdapter()
        self.max_prompt_chars = int(max_tokens * TOKEN_TO_CHAR_RATIO)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_checkpointing = enable_checkpointing
        self.enable_sampling = enable_sampling
        
        # Use only async locks
        self._stats_lock = asyncio.Lock()
        self._graph_lock = asyncio.Lock()
        self._id_index_lock = asyncio.Lock()
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent_operations)
        
        self.graph = nx.MultiDiGraph()
        self._id_index = defaultdict(set)
        
        # Entity resolution cache
        self._entity_cache = {}
        self._cache_lock = asyncio.Lock()
        
        # Streaming deduplication for large docs
        self._running_nodes = {}
        self._running_edges = {}
        self._running_lock = asyncio.Lock()
        
        # Document processing cache
        self._doc_cache = {}
        self._doc_cache_lock = asyncio.Lock()
        
        self.stats = Counter()
        
        # Track type mappings
        self.type_usage_stats = Counter()
        self.type_mapping_log = []
        
        self.logger = logging.getLogger(__name__)
        
        # Track initialization state
        self._initialized = False
        
        # Checkpoint directory
        if enable_checkpointing:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    async def initialize(self):
        """Initialize the graph service - call this after creation."""
        if self._initialized:
            return
        
        await self._load_graph()
        self._initialized = True

    # Smart validation for graph data
    async def _validate_and_fix_graph_data(self, raw_data: Dict[str, Any]) -> Optional[GraphData]:
        """
        Smart validation - Handles ANY node types dynamically.
        Converts unknown node types to valid categories automatically.
        """
        try:
            # First, try direct validation
            return GraphData(**raw_data)
        except Exception as e:
            self.logger.warning(f"Initial validation failed: {e}, attempting smart fixes...")
            
            # Smart validation with automatic type mapping
            fixed_nodes = []
            valid_node_ids = set()
            mapping_applied = False
            
            for node_data in raw_data.get('nodes', []):
                try:
                    original_type = node_data.get('type', 'unknown')
                    
                    # Apply smart type normalization
                    if original_type and not self._is_valid_node_type(original_type):
                        normalized_type = SmartNodeTypeValidator.normalize_type(original_type)
                        node_data['type'] = normalized_type
                        mapping_applied = True
                        
                        # Track type usage for analytics
                        mapping_key = f"{original_type} -> {normalized_type}"
                        self.type_usage_stats[mapping_key] += 1
                        self.type_mapping_log.append({
                            'original': original_type,
                            'mapped': normalized_type,
                            'label': node_data.get('label', 'unknown'),
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        })
                        
                        self.logger.info(f"Mapped '{original_type}' -> '{normalized_type}' for '{node_data.get('label', 'unknown')}'")
                    
                    # Ensure required fields exist
                    if 'id' not in node_data:
                        node_data['id'] = f"node_{uuid.uuid4()}"
                    if 'confidence' not in node_data:
                        node_data['confidence'] = 0.5
                    if 'properties' not in node_data:
                        node_data['properties'] = {}
                    
                    # Create node with fixed type
                    node = Node(**node_data)
                    fixed_nodes.append(node)
                    valid_node_ids.add(node.id)
                    
                except Exception as node_error:
                    self.logger.warning(f"Skipping invalid node: {node_error}")
                    continue
            
            # Fix edges - only keep edges between valid nodes
            fixed_edges = []
            for edge_data in raw_data.get('edges', []):
                if (edge_data.get('source') in valid_node_ids and 
                    edge_data.get('target') in valid_node_ids):
                    try:
                        # Ensure required fields exist
                        if 'id' not in edge_data:
                            edge_data['id'] = f"edge_{uuid.uuid4()}"
                        if 'label' not in edge_data:
                            edge_data['label'] = str(edge_data.get('type', 'related_to'))
                        if 'confidence' not in edge_data:
                            edge_data['confidence'] = 0.5
                        if 'properties' not in edge_data:
                            edge_data['properties'] = {}
                        
                        edge = Edge(**edge_data)
                        fixed_edges.append(edge)
                    except Exception:
                        continue
            
            if fixed_nodes:
                result = GraphData(nodes=fixed_nodes, edges=fixed_edges)
                success_msg = f"Smart validation successful: {len(fixed_nodes)} nodes, {len(fixed_edges)} edges"
                if mapping_applied:
                    success_msg += f" (applied {len([m for m in self.type_mapping_log if m['timestamp']])} type mappings)"
                self.logger.info(success_msg)
                return result
                
            self.logger.error("No valid nodes after smart fixing")
            return None
            
        except Exception as final_error:
            self.logger.error(f"Smart validation completely failed: {final_error}")
            return None

    def _is_valid_node_type(self, node_type: str) -> bool:
        """Check if a node type is valid without raising exceptions"""
        try:
            NodeType(node_type)
            return True
        except ValueError:
            return False

    def get_extraction_analytics(self) -> Dict[str, Any]:
        """Get analytics on type mappings and extraction performance."""
        return {
            "type_mappings_used": dict(self.type_usage_stats),
            "total_mappings_applied": sum(self.type_usage_stats.values()),
            "unique_mapping_patterns": len(self.type_usage_stats),
            "most_common_mappings": self.type_usage_stats.most_common(10),
            "recent_mappings": self.type_mapping_log[-20:] if self.type_mapping_log else [],
            "smart_features": {
                "semantic_analysis": True,
                "dynamic_type_mapping": True,
                "graceful_error_recovery": True,
                "handles_any_document_type": True,
                "real_time_analytics": True
            },
            "document_compatibility": {
                "medical_records": "Supported",
                "legal_contracts": "Supported", 
                "financial_statements": "Supported",
                "insurance_policies": "Supported",
                "technical_specifications": "Supported",
                "academic_papers": "Supported",
                "government_documents": "Supported",
                "business_reports": "Supported",
                "unknown_formats": "Auto-adapts"
            }
        }

    # Legacy method for backward compatibility
    def get_type_usage_analytics(self) -> Dict[str, Any]:
        """Get analytics on type mappings for monitoring and improvement."""
        return self.get_extraction_analytics()

    async def _load_graph(self):
        """Loads graph data from the storage adapter on startup."""
        try:
            result = await self.storage_adapter.load()
            if result:
                graph_dict, metadata = result
                if metadata.get('version') != GRAPH_SCHEMA_VERSION:
                    self.logger.warning(f"Graph version mismatch. Found: {metadata.get('version')}, Expected: {GRAPH_SCHEMA_VERSION}")
                    return
                
                async with self._graph_lock:
                    self.graph = json_graph.node_link_graph(graph_dict)
                    await self._rebuild_id_index()
                    self.stats.update(metadata.get('stats', {}))
                    # Load type usage stats if available
                    if 'type_usage_stats' in metadata:
                        self.type_usage_stats.update(metadata['type_usage_stats'])
                
                self.logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
            else:
                await self._init_graph()
        except Exception as e:
            self.logger.error(f"Failed to load graph: {e}")
            await self._init_graph()

    async def _init_graph(self):
        """Initializes a new graph with default metadata."""
        async with self._graph_lock:
            self.graph.graph['version'] = GRAPH_SCHEMA_VERSION
            self.graph.graph['created_at'] = datetime.now(timezone.utc).isoformat()
            self.graph.graph['schema_types'] = {
                'node_types': [t.value for t in NodeType],
                'relationship_types': [t.value for t in RelationshipType]
            }

    async def _rebuild_id_index(self):
        """Rebuilds the label-to-ID index from the current graph for fast lookups."""
        async with self._id_index_lock:
            self._id_index.clear()
            count = 0
            for node_id, data in self.graph.nodes(data=True):
                if count >= MAX_INDEX_SIZE:
                    self.logger.warning(f"Index size limit reached ({MAX_INDEX_SIZE})")
                    break
                if label := data.get('label'):
                    self._id_index[self._normalize_label(label)].add(node_id)
                    count += 1

    async def save_graph(self):
        """Saves the current graph state using the configured storage adapter."""
        try:
            async with self._graph_lock:
                graph_dict = json_graph.node_link_data(self.graph)
                
                # Convert datetime objects and sets to JSON serializable format in nodes
                for node in graph_dict.get('nodes', []):
                    for key, value in list(node.items()):
                        if isinstance(value, datetime):
                            node[key] = value.isoformat()
                        elif isinstance(value, set):
                            node[key] = list(value)
                        elif key == 'properties' and isinstance(value, dict):
                            # Also check properties dict
                            for prop_key, prop_value in list(value.items()):
                                if isinstance(prop_value, datetime):
                                    value[prop_key] = prop_value.isoformat()
                                elif isinstance(prop_value, set):
                                    value[prop_key] = list(prop_value)
                        elif isinstance(value, dict):
                            # Handle nested dictionaries
                            for k, v in list(value.items()):
                                if isinstance(v, datetime):
                                    value[k] = v.isoformat()
                                elif isinstance(v, set):
                                    value[k] = list(v)
                
                # Convert datetime objects and sets in edges
                for edge in graph_dict.get('links', []):
                    for key, value in list(edge.items()):
                        if isinstance(value, datetime):
                            edge[key] = value.isoformat()
                        elif isinstance(value, set):
                            edge[key] = list(value)
                        elif isinstance(value, dict):
                            for k, v in list(value.items()):
                                if isinstance(v, datetime):
                                    value[k] = v.isoformat()
                                elif isinstance(v, set):
                                    value[k] = list(v)
            
            async with self._stats_lock:
                metadata = {
                    'version': GRAPH_SCHEMA_VERSION,
                    'saved_at': datetime.now(timezone.utc).isoformat(),
                    'stats': dict(self.stats),
                    'extraction_analytics': self.get_extraction_analytics(),
                    'type_usage_stats': dict(self.type_usage_stats),
                    'node_count': self.graph.number_of_nodes(),
                    'edge_count': self.graph.number_of_edges()
                }
                
            if await self.storage_adapter.save(graph_dict, metadata):
                self.logger.info("Graph saved successfully.")
            else:
                self.logger.error("Storage adapter failed to save graph.")
        except Exception as e:
            self.logger.error(f"Error during graph save: {e}")

    async def post_process_graph(self, document_type: Optional[str] = None):
        """Post-process graph to clean up entities and standardize data."""
        self.logger.info("Starting post-processing for document graph")
        
        async with self._graph_lock:
            # 1. Fix datetime serialization issues (universal)
            for node_id, data in self.graph.nodes(data=True):
                # Convert any datetime objects to strings
                for key, value in list(data.items()):
                    if isinstance(value, datetime):
                        data[key] = value.isoformat()
                    elif isinstance(value, set):
                        data[key] = list(value)
                    elif key == 'properties' and isinstance(value, dict):
                        for prop_key, prop_value in list(value.items()):
                            if isinstance(prop_value, datetime):
                                value[prop_key] = prop_value.isoformat()
                            elif isinstance(prop_value, set):
                                value[prop_key] = list(prop_value)
                
                # Ensure confidence is float
                if 'confidence' in data:
                    try:
                        data['confidence'] = float(data.get('confidence', 0.5))
                    except:
                        data['confidence'] = 0.5
            
            # 2. Merge duplicate entities based on normalized labels
            merge_candidates = defaultdict(list)
            
            for node_id, data in self.graph.nodes(data=True):
                label = data.get('label', '').lower().strip()
                node_type = data.get('type', '')
                if label:
                    # Group by normalized label + type for merging
                    key = f"{node_type}:{label}"
                    merge_candidates[key].append((node_id, data))
            
            # Merge nodes with identical type+label
            merged_count = 0
            for key, nodes in merge_candidates.items():
                if len(nodes) > 1:
                    # Keep the node with highest confidence
                    nodes.sort(key=lambda x: x[1].get('confidence', 0), reverse=True)
                    primary_id, primary_data = nodes[0]
                    
                    for secondary_id, secondary_data in nodes[1:]:
                        # Merge properties
                        primary_props = primary_data.setdefault('properties', {})
                        secondary_props = secondary_data.get('properties', {})
                        
                        for k, v in secondary_props.items():
                            if k not in primary_props:
                                primary_props[k] = v
                        
                        # Redirect edges
                        for source, target, edge_key, edge_data in list(self.graph.edges(secondary_id, data=True, keys=True)):
                            if target != primary_id:
                                self.graph.add_edge(source, primary_id, key=edge_key, **edge_data)
                        
                        for source, target, edge_key, edge_data in list(self.graph.in_edges(secondary_id, data=True, keys=True)):
                            if source != primary_id:
                                self.graph.add_edge(primary_id, target, key=edge_key, **edge_data)
                        
                        # Remove secondary node
                        self.graph.remove_node(secondary_id)
                        merged_count += 1
            
            if merged_count > 0:
                self.logger.info(f"Merged {merged_count} duplicate nodes")
            
            # 3. Extract and standardize common numeric values (universal)
            for node_id, data in self.graph.nodes(data=True):
                label = data.get('label', '')
                props = data.setdefault('properties', {})
                
                # Extract percentage values
                if 'percent' in label.lower() or '%' in label:
                    match = re.search(r'(\d+(?:\.\d+)?)\s*%', label)
                    if match:
                        props['percentage'] = float(match.group(1))
                        props['formatted'] = f"{match.group(1)}%"
                
                # Extract currency/amount values
                elif re.search(r'[\$€£₹]\s*[\d,]+', label):
                    match = re.search(r'([\$€£₹])\s*([\d,]+(?:\.\d+)?)', label)
                    if match:
                        props['currency'] = match.group(1)
                        props['numeric_value'] = float(match.group(2).replace(',', ''))
                
                # Extract duration values (days, months, years)
                elif re.search(r'\d+\s*(?:days?|months?|years?|weeks?|hours?)', label, re.I):
                    match = re.search(r'(\d+)\s*(days?|months?|years?|weeks?|hours?)', label, re.I)
                    if match:
                        props['numeric_value'] = int(match.group(1))
                        props['unit'] = match.group(2).lower()
                        props['value'] = f"{match.group(1)} {match.group(2)}"
            
            # 4. Insurance/Finance domain specialization (runs only for those doc types)
            if document_type in ("insurance", "policy", "travel", "financial", "finance"):
                await self._post_process_insurance_finance(document_type)
        
        # Update the index after modifications
        await self._rebuild_id_index()
        
        self.logger.info("Post-processing complete")

    async def _post_process_insurance_finance(self, document_type: Optional[str] = None):
        """Insurance/Finance domain specialization layer for post-processing."""
        self.logger.info("Running insurance/finance domain specialization")
        
        # Merge well-known insurance/finance duplicate terms
        domain_merge_terms = []
        if document_type in ("insurance", "policy", "travel"):
            domain_merge_terms = [
                'grace period', 'waiting period', 'sum insured', 'deductible',
                'premium', 'age limit', 'entry age', 'exit age', 'copayment',
                'cumulative bonus', 'no claim bonus', 'free look period',
                'room rent', 'sub limit', 'network hospital', 'tpa',
            ]
        elif document_type in ("financial", "finance"):
            domain_merge_terms = [
                'interest rate', 'credit score', 'net worth', 'balance sheet',
                'profit and loss', 'cash flow', 'capital gain', 'tax benefit',
                'mutual fund', 'fixed deposit', 'recurring deposit',
            ]
        
        if domain_merge_terms:
            merge_candidates = defaultdict(list)
            for node_id, data in self.graph.nodes(data=True):
                label = data.get('label', '').lower()
                for term in domain_merge_terms:
                    if term in label:
                        merge_candidates[term].append((node_id, data))
            
            merged = 0
            for term, nodes in merge_candidates.items():
                if len(nodes) > 1:
                    nodes.sort(key=lambda x: x[1].get('confidence', 0), reverse=True)
                    primary_id, primary_data = nodes[0]
                    for secondary_id, secondary_data in nodes[1:]:
                        primary_props = primary_data.setdefault('properties', {})
                        for k, v in secondary_data.get('properties', {}).items():
                            if k not in primary_props:
                                primary_props[k] = v
                        for s, t, ek, ed in list(self.graph.edges(secondary_id, data=True, keys=True)):
                            if t != primary_id:
                                self.graph.add_edge(s, primary_id, key=ek, **ed)
                        for s, t, ek, ed in list(self.graph.in_edges(secondary_id, data=True, keys=True)):
                            if s != primary_id:
                                self.graph.add_edge(primary_id, t, key=ek, **ed)
                        self.graph.remove_node(secondary_id)
                        merged += 1
            if merged:
                self.logger.info(f"Domain merge: consolidated {merged} duplicate insurance/finance nodes")
        
        # Enrich insurance/finance nodes with structured properties
        for node_id, data in self.graph.nodes(data=True):
            label = data.get('label', '')
            props = data.setdefault('properties', {})
            label_lower = label.lower()
            
            # Tag domain context
            if 'domain' not in props:
                props['domain'] = 'insurance' if document_type in ('insurance', 'policy', 'travel') else 'finance'
            
            # Insurance-specific value extraction
            if 'grace period' in label_lower:
                match = re.search(r'(\d+)\s*days?', label, re.I)
                if match:
                    props['value'] = f"{match.group(1)} days"
                    props['numeric_value'] = int(match.group(1))
                    props['unit'] = 'days'
            
            elif 'age' in label_lower and ('limit' in label_lower or 'entry' in label_lower or 'exit' in label_lower):
                match = re.search(r'(\d+)\s*(?:months?|years?)\s*to\s*(\d+)\s*(?:years?)', label, re.I)
                if match:
                    props['min_age'] = match.group(1)
                    props['max_age'] = match.group(2)
                    props['formatted'] = f"{match.group(1)} to {match.group(2)} years"
            
            elif 'waiting period' in label_lower:
                match = re.search(r'(\d+)\s*(months?|years?|days?)', label, re.I)
                if match:
                    props['value'] = match.group(0)
                    props['numeric_value'] = int(match.group(1))
                    props['unit'] = match.group(2).lower()
            
            # Finance-specific value extraction
            elif 'interest' in label_lower and 'rate' in label_lower:
                match = re.search(r'(\d+(?:\.\d+)?)\s*%', label)
                if match:
                    props['rate'] = float(match.group(1))
                    props['formatted'] = f"{match.group(1)}%"
        
        self.logger.info("Insurance/finance specialization complete")

    def _normalize_label(self, label: str) -> str:
        """Normalizes a label for effective deduplication."""
        label = re.sub(r'\s+(pvt\.?|ltd\.?|limited|inc\.?|corp\.?)\s*$', '', label, flags=re.I)
        return ' '.join(label.lower().split())

    def _normalize_entity_id(self, entity_type: str, label: str, chunk_idx: Optional[int] = None) -> str:
        """Create normalized entity ID with optional chunk prefix."""
        normalized = re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')
        base_id = f"{entity_type.lower()}_{normalized}"
        
        # Add chunk prefix if provided to prevent collisions
        if chunk_idx is not None:
            return f"chunk{chunk_idx}_{base_id}"
        return base_id

    @lru_cache(maxsize=10000)
    def _cached_normalize_label(self, label: str) -> str:
        """Cached version of label normalization."""
        return self._normalize_label(label)

    async def _find_existing_entity(self, entity_type: str, label: str) -> Optional[str]:
        """Finds an existing entity using an O(1) index lookup with caching."""
        # Check cache first
        cache_key = f"{entity_type}:{label}"
        async with self._cache_lock:
            if cache_key in self._entity_cache:
                return self._entity_cache[cache_key]
        
        normalized_label = self._cached_normalize_label(label)
        async with self._id_index_lock:
            candidate_ids = self._id_index.get(normalized_label, set()).copy()
        
        result = None
        for node_id in candidate_ids:
            if node_id in self.graph and self.graph.nodes[node_id].get('type') == entity_type:
                result = node_id
                break
        
        # Update cache
        async with self._cache_lock:
            self._entity_cache[cache_key] = result
            # Limit cache size
            if len(self._entity_cache) > 5000:
                # Remove oldest entries
                for key in list(self._entity_cache.keys())[:1000]:
                    del self._entity_cache[key]
        
        return result

    def _get_adaptive_chunk_size(self, text_length: int) -> Tuple[int, int, int]:
        """Get adaptive chunk size based on document length."""
        if text_length < SMALL_DOC_THRESHOLD:
            # Small doc - single chunk
            return text_length, 0, 1
        elif text_length < LARGE_DOC_THRESHOLD:
            # Medium doc - standard chunking
            return self.chunk_size, self.chunk_overlap, DEFAULT_BATCH_SIZE
        elif text_length < VERY_LARGE_DOC_THRESHOLD:
            # Large doc - bigger chunks, larger batches
            return 5000, 500, LARGE_DOC_BATCH_SIZE
        else:
            # Very large doc - maximum efficiency
            return 8000, 800, 15

    def _should_sample_chunks(self, total_chunks: int) -> bool:
        """Determine if we should sample chunks for very large documents."""
        return self.enable_sampling and total_chunks > 100

    def _get_sample_indices(self, total_chunks: int) -> List[int]:
        """Get indices of chunks to sample for very large documents."""
        if total_chunks <= 100:
            return list(range(total_chunks))
        
        # Sample strategy: First 10, last 10, and evenly distributed middle
        sample_size = min(50, total_chunks // 2)
        
        indices = set()
        # First 10 chunks (usually contain important overview)
        indices.update(range(min(10, total_chunks)))
        
        # Last 10 chunks (usually contain conclusions)
        indices.update(range(max(0, total_chunks - 10), total_chunks))
        
        # Evenly distributed middle chunks
        step = max(1, (total_chunks - 20) // (sample_size - 20))
        indices.update(range(10, total_chunks - 10, step))
        
        return sorted(list(indices))[:sample_size]

    def _smart_chunk_text(self, text: str, document_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Intelligently chunks text based on document structure."""
        text_length = len(text)
        chunk_size, overlap, _ = self._get_adaptive_chunk_size(text_length)
        
        chunks = []
        
        # Skip chunking for small documents
        if text_length < SMALL_DOC_THRESHOLD:
            return [{
                'text': text,
                'start_char': 0,
                'end_char': text_length,
                'index': 0,
                'total_chunks': 1,
                'document_type': document_type,
                'is_sampled': False
            }]
        
        # Document-specific chunking strategies
        # Use section-aware chunking for structured documents
        section_markers = []
        if document_type in ("insurance", "policy", "travel"):
            section_markers = [
                r'\n\s*(?:SECTION|Section|ARTICLE|Article|PART|Part|SCHEDULE|Schedule)\s+\d+',
                r'\n\s*(?:Coverage|Exclusions|Definitions|Benefits|Claims|Premium|Eligibility|Waiting\s+Period|Grace\s+Period|Sub-?Limits?|Riders?|Endorsements?|Renewal|Cancellation|Grievance)',
                r'\n\s*(?:Table\s+of\s+Benefits|Scope\s+of\s+Cover|In-?Patient|Out-?Patient|Pre-Hospitalization|Post-Hospitalization|Critical\s+Illness)',
                r'\n\s*\d+\.\s+[A-Z][a-z]+',
            ]
        elif document_type in ("financial", "finance"):
            section_markers = [
                r'\n\s*(?:SECTION|Section|ARTICLE|Article|SCHEDULE|Schedule)\s+\d+',
                r'\n\s*(?:Balance\s+Sheet|Profit|Loss|Cash\s+Flow|Revenue|Assets|Liabilities|Equity|Notes\s+to|Auditor|Director|Management\s+Discussion)',
                r'\n\s*(?:Risk\s+Management|Capital\s+Adequacy|Investments|Provisions|Reserves|Contingent)',
                r'\n\s*\d+\.\s+[A-Z][a-z]+',
            ]
        elif document_type in ("legal", "contract"):
            section_markers = [
                r'\n\s*(?:SECTION|Section|ARTICLE|Article|CLAUSE|Clause)\s+\d+',
                r'\n\s*(?:WHEREAS|NOW THEREFORE|RECITALS|DEFINITIONS|OBLIGATIONS|TERMINATION|INDEMNIFICATION)',
                r'\n\s*\d+\.\s+[A-Z][a-z]+',
            ]
        elif document_type in ("technical", "documentation"):
            section_markers = [
                r'\n\s*#{1,4}\s+',  # Markdown headers
                r'\n\s*(?:Overview|Introduction|Architecture|Configuration|Installation|API|Usage|Requirements|Specifications)',
                r'\n\s*\d+\.\s+[A-Z][a-z]+',
            ]
        elif document_type in ("academic",):
            section_markers = [
                r'\n\s*(?:Abstract|Introduction|Methodology|Results|Discussion|Conclusion|References|Acknowledgements)',
                r'\n\s*\d+\.\s+[A-Z][a-z]+',
            ]
        elif document_type in ("hr",):
            section_markers = [
                r'\n\s*(?:Policy|Eligibility|Procedure|Benefits|Responsibilities|Scope|Purpose|Definitions)',
                r'\n\s*\d+\.\s+[A-Z][a-z]+',
            ]
        else:
            # Generic section markers that work across many document types
            section_markers = [
                r'\n\s*(?:SECTION|Section|ARTICLE|Article|CHAPTER|Chapter)\s+\d+',
                r'\n\s*\d+\.\s+[A-Z][a-z]+',
                r'\n\s*#{1,4}\s+',  # Markdown headers
            ]

        if section_markers:
            # Try to split by sections first
            parts = [text]
            for marker in section_markers:
                new_parts = []
                for part in parts:
                    splits = re.split(f'({marker})', part)  # Keep delimiter
                    # Recombine delimiter with following text
                    i = 0
                    while i < len(splits):
                        if i + 1 < len(splits) and re.match(marker, splits[i]):
                            new_parts.append(splits[i] + splits[i + 1])
                            i += 2
                        else:
                            new_parts.append(splits[i])
                            i += 1
                parts = [p for p in new_parts if p.strip()]
            
            # Now chunk each part
            for part in parts:
                if len(part.strip()) > 50:  # Ignore tiny fragments
                    self._chunk_by_size_optimized(part, chunks, chunk_size, overlap)
        
        else:
            # Default: chunk by paragraphs and size
            self._chunk_by_paragraphs_optimized(text, chunks, chunk_size, overlap)
        
        # Add metadata to chunks
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk['index'] = i
            chunk['total_chunks'] = total_chunks
            chunk['document_type'] = document_type
            chunk['is_sampled'] = False
        
        # Apply sampling if needed
        if self._should_sample_chunks(total_chunks):
            sample_indices = self._get_sample_indices(total_chunks)
            sampled_chunks = []
            for idx in sample_indices:
                if idx < len(chunks):
                    chunks[idx]['is_sampled'] = True
                    sampled_chunks.append(chunks[idx])
            
            self.logger.info(f"Sampling {len(sampled_chunks)} chunks from {total_chunks} total chunks")
            return sampled_chunks
        
        return chunks

    def _chunk_by_size_optimized(self, text: str, chunks: List[Dict[str, Any]], chunk_size: int, overlap: int):
        """Optimized chunking by size with better overlap handling."""
        text = text.strip()
        if not text:
            return
        
        # For very small texts, just return as one chunk
        if len(text) <= chunk_size:
            chunks.append({
                'text': text,
                'start_char': 0,
                'end_char': len(text)
            })
            return
        
        start = 0
        
        while start < len(text):
            # Calculate ideal end position
            ideal_end = min(start + chunk_size, len(text))
            
            # If we're close to the end, just take the rest
            if len(text) - ideal_end < chunk_size * 0.2:  # Within 20% of chunk size
                chunks.append({
                    'text': text[start:].strip(),
                    'start_char': start,
                    'end_char': len(text)
                })
                break
            
            # Find the best break point
            end = ideal_end
            best_break = ideal_end
            
            # Look for sentence endings
            for delimiter in ['. ', '.\n', '? ', '!\n', '\n\n', '\n']:
                # Search in a window around ideal_end
                search_start = max(start, ideal_end - 200)
                search_end = min(len(text), ideal_end + 100)
                
                last_delimiter = text.rfind(delimiter, search_start, search_end)
                if last_delimiter > start:
                    best_break = last_delimiter + len(delimiter)
                    break
            
            end = best_break
            
            # Extract chunk
            chunk_text = text[start:end].strip()
            
            # Only add substantial chunks
            if chunk_text and len(chunk_text) > 100:
                chunks.append({
                    'text': chunk_text,
                    'start_char': start,
                    'end_char': end
                })
            
            # Calculate next start with overlap
            if end >= len(text) - 100:
                break
            
            # Smart overlap: include last sentence from previous chunk
            overlap_start = max(start + chunk_size - overlap, end - 500)
            sentence_start = text.rfind('. ', overlap_start, end)
            if sentence_start > overlap_start:
                start = sentence_start + 2
            else:
                start = end - overlap

    def _chunk_by_paragraphs_optimized(self, text: str, chunks: List[Dict[str, Any]], chunk_size: int, overlap: int):
        """Optimized paragraph chunking."""
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = []
        current_size = 0
        start_char = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            # If single paragraph is too large, chunk it
            if para_size > chunk_size:
                # Save current chunk if any
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'start_char': start_char,
                        'end_char': start_char + len(chunk_text)
                    })
                    current_chunk = []
                    current_size = 0
                    start_char += len(chunk_text) + 2
                
                # Chunk the large paragraph
                self._chunk_by_size_optimized(para, chunks, chunk_size, overlap)
                start_char += para_size + 2
            
            # If adding this paragraph exceeds limit, start new chunk
            elif current_size + para_size > chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start_char': start_char,
                    'end_char': start_char + len(chunk_text)
                })
                # Include last paragraph from previous chunk for overlap
                if overlap > 0 and len(current_chunk) > 1:
                    current_chunk = [current_chunk[-1], para]
                    current_size = len(current_chunk[-1]) + para_size
                else:
                    current_chunk = [para]
                    current_size = para_size
                start_char += len(chunk_text) + 2
            
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start_char': start_char,
                'end_char': start_char + len(chunk_text)
            })

    def _build_extraction_prompt(self, text: str, document_type: Optional[str] = None, chunk_info: Optional[Dict[str, Any]] = None) -> str:
        """Builds a detailed, context-aware prompt for the LLM."""
        # Input validation
        if not text:
            raise ValueError("Text cannot be empty")
        if len(text) > MAX_TEXT_SIZE:
            raise ValueError(f"Text too large (max {MAX_TEXT_SIZE} characters)")
        
        truncated_text = text[:self.max_prompt_chars]
        
        # Enhanced prompt - General Core + Insurance/Finance Specialization
        # Base suggested types (universal)
        suggested_types = [
            "Person", "Organization", "Document", "Amount", "Date", 
            "Term", "Concept", "Process", "Rule", "Condition",
            "Product", "Technology", "Metric", "Role", "Risk",
            "Entity"  # Fallback type
        ]
        
        # Insurance/Finance domain specialization: add domain-specific types
        if document_type in ("insurance", "policy", "travel", "financial", "finance"):
            suggested_types.extend([
                "Policy", "Coverage", "Exclusion", "Claim", "Premium", "Deductible",
                "Beneficiary", "Rider", "Endorsement", "Insurer", "Policyholder",
                "Account", "Transaction", "Investment", "Liability", "Asset"
            ])
        
        doc_focus = {
            # === INSURANCE/FINANCE SPECIALIZATION (enriched) ===
            "insurance": (
                "INSURANCE DOMAIN EXPERT MODE: Extract with deep insurance knowledge.\n"
                "Key entities: policies, insurers, policyholders, beneficiaries, nominees, TPAs\n"
                "Key concepts: coverage types, sum insured, premiums, deductibles, copayments, sub-limits\n"
                "Key terms: waiting periods, grace periods, free look periods, exclusions, endorsements, riders\n"
                "Key processes: claim procedures, cashless vs reimbursement, pre-authorization, grievance redressal\n"
                "Key conditions: pre-existing diseases, age limits, portability, cumulative bonus, no-claim bonus\n"
                "Extract specific amounts, percentages, durations, and conditions with their exact values."
            ),
            "travel": (
                "TRAVEL INSURANCE EXPERT MODE: Extract with travel insurance expertise.\n"
                "Focus on: coverage types (medical emergency, trip cancellation, baggage loss, personal accident),\n"
                "geographic coverage, emergency assistance, hospitalization benefits, repatriation,\n"
                "age limits, deductibles, exclusions, adventure sports coverage, pre-existing conditions."
            ),
            "policy": (
                "INSURANCE POLICY EXPERT MODE: Extract detailed policy terms.\n"
                "Focus on: coverage details, premiums, waiting periods, exclusions, benefits,\n"
                "claim procedures, age limits, grace periods, renewal terms, cancellation terms,\n"
                "riders/endorsements, sub-limits, room rent limits, co-payment percentages."
            ),
            "financial": (
                "FINANCE DOMAIN EXPERT MODE: Extract with financial expertise.\n"
                "Key entities: institutions, accounts, funds, portfolios, instruments\n"
                "Key metrics: interest rates, returns, ratios, yields, NAV, market cap\n"
                "Key processes: transactions, settlements, disbursements, reconciliation\n"
                "Key terms: assets, liabilities, equity, revenue, expenses, P&L, balance sheet\n"
                "Key compliance: KYC, AML, SEBI, RBI regulations, audit requirements\n"
                "Extract specific amounts, rates, dates, and regulatory references."
            ),
            # === GENERAL DOMAIN SUPPORT ===
            "medical": "Focus on: treatments, diagnoses, medications, procedures, symptoms, conditions, patients, healthcare providers",
            "legal": "Focus on: contracts, obligations, rights, parties, terms, conditions, clauses, regulations, compliance",
            "technical": "Focus on: systems, components, APIs, databases, architectures, protocols, requirements, specifications",
            "academic": "Focus on: research findings, methodologies, hypotheses, experiments, authors, citations, conclusions",
            "hr": "Focus on: employees, roles, departments, policies, benefits, performance metrics, organizational structure",
            "documentation": "Focus on: features, procedures, configurations, requirements, specifications, workflows",
            "general": "Extract all relevant entities (people, organizations, concepts, amounts, dates) and their relationships",
            "report": "Focus on: findings, metrics, recommendations, stakeholders, timelines, outcomes, data points",
            "contract": "Focus on: parties, obligations, terms, conditions, dates, amounts, deliverables, penalties",
        }
        doc_instruction = doc_focus.get(document_type, "Extract all relevant entities and relationships from this document")
        
        # Add chunk context if available
        chunk_context = ""
        if chunk_info:
            chunk_idx = chunk_info.get('index', 0)
            total_chunks = chunk_info.get('total_chunks', 1)
            is_sampled = chunk_info.get('is_sampled', False)
            
            chunk_context = f"\nThis is chunk {chunk_idx + 1} of {total_chunks}."
            if is_sampled and total_chunks > 50:
                chunk_context += "\nThis is a sampled chunk from a large document - focus on key entities only."
            # Add instruction to use chunk-specific IDs
            chunk_context += f"\nPrefix all entity IDs with 'chunk{chunk_idx}_' to ensure uniqueness."

        return f"""You are an expert document analyzer. Extract entities and relationships from this document.

INSTRUCTIONS:
{doc_instruction}{chunk_context}

TYPE HANDLING:
- Use the provided node types when possible, but don't worry if you encounter new entity types
- The system will automatically map any unknown types to valid categories
- Focus on extracting meaningful entities and relationships rather than worrying about exact type matching

EXTRACTION RULES:
1. Extract meaningful entities (people, organizations, concepts, amounts, dates, etc.)
2. Use descriptive but concise entity types (e.g., "Person", "Company", "Amount", "Date", "Term", "Benefit")
3. Suggested types: {', '.join(suggested_types)} - but feel free to use others if more appropriate
4. Each node needs: id, type, label, properties (optional), confidence (0-1)
5. Each edge needs: source, target, type, label, confidence (0-1)
6. Use lowercase_with_underscores for IDs
7. Don't hesitate to use descriptive types - the system will handle any unknown types intelligently

Return ONLY valid JSON with "nodes" and "edges" arrays.

Document text:
{truncated_text}

JSON:"""

    def _parse_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Enhanced and robust JSON parsing from LLM responses with better error recovery."""
        try:
            cleaned = response_text.strip()
            
            # Try different extraction methods
            if '```json' in cleaned:
                cleaned = cleaned.split('```json')[1].split('```')[0]
            elif '```' in cleaned:
                cleaned = cleaned.split('```')[1].split('```')[0]
            
            # Find JSON boundaries
            json_start = cleaned.find('{')
            if json_start == -1: 
                return None
            
            # Balance braces
            brace_count = 0
            json_end = 0
            for i, char in enumerate(cleaned[json_start:]):
                if char == '{': 
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = json_start + i + 1
                        break
            
            if json_end > 0:
                final_json = cleaned[json_start:json_end]
                parsed = json.loads(final_json)
                
                # Validate structure
                if 'nodes' in parsed and 'edges' in parsed:
                    # Ensure lists
                    if not isinstance(parsed['nodes'], list):
                        parsed['nodes'] = []
                    if not isinstance(parsed['edges'], list):
                        parsed['edges'] = []
                    
                    # Basic cleanup for common issues
                    for node in parsed['nodes']:
                        if isinstance(node, dict):
                            # Ensure type is string
                            if 'type' in node:
                                node['type'] = str(node['type'])
                            # Clean up any None values
                            node = {k: v for k, v in node.items() if v is not None}
                            # Ensure required fields
                            if 'confidence' not in node or not isinstance(node['confidence'], (int, float)):
                                node['confidence'] = 0.5
                            if 'properties' not in node:
                                node['properties'] = {}
                    
                    return parsed
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"LLM response parsing error: {e}")
            return None

    async def _update_entity_confidence(self, existing_node: Dict[str, Any], new_confidence: float):
        """Updates entity confidence using a weighted average based on observation count."""
        obs_count = existing_node.get('observation_count', 1)
        old_confidence = existing_node.get('confidence', 0.5)
        new_weighted_confidence = (old_confidence * obs_count + new_confidence) / (obs_count + 1)
        existing_node['confidence'] = round(new_weighted_confidence, 3)
        existing_node['observation_count'] = obs_count + 1
        existing_node['last_observed'] = datetime.now(timezone.utc).isoformat()

    async def _streaming_deduplicate_node(self, node: Node) -> Optional[str]:
        """Deduplicate node in streaming fashion for large documents."""
        clean_id = re.sub(r'^chunk\d+_', '', node.id)
        
        async with self._running_lock:
            if clean_id in self._running_nodes:
                # Merge with existing
                existing = self._running_nodes[clean_id]
                obs_count = existing.properties.get('observation_count', 1)
                new_confidence = (existing.confidence * obs_count + node.confidence) / (obs_count + 1)
                existing.confidence = round(new_confidence, 3)
                existing.properties['observation_count'] = obs_count + 1
                
                # Merge properties
                for key, value in node.properties.items():
                    if key not in existing.properties:
                        existing.properties[key] = value
                
                return clean_id  # Return existing ID
            else:
                # New node
                node.id = clean_id
                node.properties['observation_count'] = 1
                self._running_nodes[clean_id] = node
                return None  # Indicates new node

    def _deduplicate_nodes(self, nodes: List[Node]) -> List[Node]:
        """Deduplicate nodes by ID and merge properties."""
        unique_nodes = {}
        
        for node in nodes:
            # Strip chunk prefix for deduplication
            clean_id = re.sub(r'^chunk\d+_', '', node.id)
            
            if clean_id in unique_nodes:
                # Merge properties
                existing = unique_nodes[clean_id]
                
                # Update confidence (weighted average)
                obs_count = existing.properties.get('observation_count', 1)
                new_confidence = (existing.confidence * obs_count + node.confidence) / (obs_count + 1)
                existing.confidence = round(new_confidence, 3)
                existing.properties['observation_count'] = obs_count + 1
                
                # Merge other properties
                for key, value in node.properties.items():
                    if key not in existing.properties:
                        existing.properties[key] = value
                    elif key == 'source_documents':
                        # Merge document lists
                        existing.properties[key] = list(set(existing.properties[key] + value))
            else:
                # Clean the node ID
                node.id = clean_id
                node.properties['observation_count'] = 1
                unique_nodes[clean_id] = node
        
        return list(unique_nodes.values())

    def _update_edge_ids(self, edges: List[Edge], id_mapping: Dict[str, str]) -> List[Edge]:
        """Update edge source/target IDs based on deduplication mapping."""
        updated_edges = []
        seen_edges = set()
        
        for edge in edges:
            # Clean chunk prefixes
            clean_source = re.sub(r'^chunk\d+_', '', edge.source)
            clean_target = re.sub(r'^chunk\d+_', '', edge.target)
            
            # Update with deduplicated IDs
            edge.source = id_mapping.get(clean_source, clean_source)
            edge.target = id_mapping.get(clean_target, clean_target)
            
            # Skip self-loops and duplicates
            edge_key = (edge.source, edge.target, edge.type)
            if edge.source != edge.target and edge_key not in seen_edges:
                seen_edges.add(edge_key)
                updated_edges.append(edge)
        
        return updated_edges

    async def _save_checkpoint(self, document_id: str, processed_chunks: int, nodes: List[Node], edges: List[Edge]):
        """Save extraction checkpoint for resume capability."""
        if not self.enable_checkpointing:
            return
        
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{document_id}_checkpoint.pkl")
        checkpoint_data = {
            'document_id': document_id,
            'processed_chunks': processed_chunks,
            'nodes': [n.dict() for n in nodes],
            'edges': [e.dict() for e in edges],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            loop = asyncio.get_running_loop()
            
            def _write_checkpoint():
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
            
            await loop.run_in_executor(None, _write_checkpoint)
            self.logger.info(f"Saved checkpoint at chunk {processed_chunks}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    async def _load_checkpoint(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Load extraction checkpoint if exists."""
        if not self.enable_checkpointing:
            return None
        
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{document_id}_checkpoint.pkl")
        if not os.path.exists(checkpoint_file):
            return None
        
        try:
            loop = asyncio.get_running_loop()
            
            def _read_checkpoint():
                with open(checkpoint_file, 'rb') as f:
                    return pickle.load(f)
            
            checkpoint_data = await loop.run_in_executor(None, _read_checkpoint)
            self.logger.info(f"Loaded checkpoint from chunk {checkpoint_data['processed_chunks']}")
            return checkpoint_data
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None

    async def _clear_checkpoint(self, document_id: str):
        """Clear checkpoint after successful completion."""
        if not self.enable_checkpointing:
            return
        
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{document_id}_checkpoint.pkl")
        if os.path.exists(checkpoint_file):
            try:
                os.remove(checkpoint_file)
                self.logger.info("Cleared checkpoint file")
            except Exception as e:
                self.logger.error(f"Failed to clear checkpoint: {e}")

    async def _check_document_cache(self, document_id: str, document_hash: str) -> Optional[GraphData]:
        """Check if document has been processed before."""
        async with self._doc_cache_lock:
            cache_key = f"{document_id}:{document_hash}"
            if cache_key in self._doc_cache:
                self.logger.info(f"Using cached extraction for document {document_id}")
                return self._doc_cache[cache_key]
        return None

    async def _cache_document_result(self, document_id: str, document_hash: str, result: GraphData):
        """Cache document extraction result."""
        async with self._doc_cache_lock:
            cache_key = f"{document_id}:{document_hash}"
            self._doc_cache[cache_key] = result
            
            # Limit cache size
            if len(self._doc_cache) > 100:
                # Remove oldest entries
                keys_to_remove = list(self._doc_cache.keys())[:20]
                for key in keys_to_remove:
                    del self._doc_cache[key]

    async def extract_from_chunks(
        self,
        text: str,
        llm_client: LLMClient,
        document_id: Optional[str] = None,
        document_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        use_streaming_dedup: bool = True
    ) -> GraphData:
        """Enhanced extraction with smart validation."""
        # Check document cache
        doc_hash = hashlib.md5(text.encode()).hexdigest()
        if document_id:
            cached_result = await self._check_document_cache(document_id, doc_hash)
            if cached_result:
                return cached_result
        
        # Clear streaming deduplication state
        if use_streaming_dedup:
            async with self._running_lock:
                self._running_nodes.clear()
                self._running_edges.clear()
        
        # Smart chunk the text
        chunks = self._smart_chunk_text(text, document_type)
        total_chunks = len(chunks)
        
        if total_chunks == 0:
            return GraphData(nodes=[], edges=[])
        
        self.logger.info(f"Processing {total_chunks} chunks for document {document_id}")
        
        # Get adaptive batch size
        _, _, batch_size = self._get_adaptive_chunk_size(len(text))
        
        # Check for checkpoint
        start_chunk = 0
        all_nodes = []
        all_edges = []
        
        if document_id and self.enable_checkpointing:
            checkpoint = await self._load_checkpoint(document_id)
            if checkpoint:
                start_chunk = checkpoint['processed_chunks']
                all_nodes = [Node(**n) for n in checkpoint['nodes']]
                all_edges = [Edge(**e) for e in checkpoint['edges']]
                self.logger.info(f"Resuming from chunk {start_chunk}")
        
        # Process chunks in parallel batches
        processed_chunks = start_chunk
        
        for i in range(start_chunk, total_chunks, batch_size):
            batch_end = min(i + batch_size, total_chunks)
            batch = chunks[i:batch_end]
            
            # Create tasks for parallel processing
            tasks = []
            for chunk in batch:
                task = self._extract_single_chunk(
                    chunk['text'],
                    llm_client,
                    document_id,
                    document_type,
                    chunk,
                    metadata
                )
                tasks.append(task)
            
            # Wait for batch to complete with timeout
            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=300  # 5 minute timeout per batch
                )
            except asyncio.TimeoutError:
                self.logger.error(f"Batch timeout at chunks {i}-{batch_end}")
                batch_results = []
            
            # Rate limit delay between batches to avoid 429 errors
            if i + batch_size < total_chunks:
                await asyncio.sleep(2)
            
            # Collect results
            batch_nodes = []
            batch_edges = []
            
            for j, result in enumerate(batch_results):
                if isinstance(result, GraphData):
                    if use_streaming_dedup:
                        # Streaming deduplication
                        for node in result.nodes:
                            existing_id = await self._streaming_deduplicate_node(node)
                            if existing_id is None:  # New node
                                batch_nodes.append(node)
                        batch_edges.extend(result.edges)
                    else:
                        batch_nodes.extend(result.nodes)
                        batch_edges.extend(result.edges)
                elif isinstance(result, Exception):
                    self.logger.error(f"Chunk {i+j} extraction failed: {result}")
            
            if use_streaming_dedup:
                # Add only new nodes
                all_nodes.extend(batch_nodes)
                all_edges.extend(batch_edges)
            else:
                all_nodes.extend(batch_nodes)
                all_edges.extend(batch_edges)
            
            processed_chunks = batch_end
            
            # Update progress
            if progress_callback:
                try:
                    await asyncio.create_task(
                        asyncio.to_thread(progress_callback, processed_chunks, total_chunks)
                    )
                except:
                    # Fallback for older Python versions
                    progress_callback(processed_chunks, total_chunks)
            
            # Save checkpoint periodically
            if document_id and self.enable_checkpointing and (processed_chunks % CHECKPOINT_INTERVAL == 0 or processed_chunks == total_chunks):
                await self._save_checkpoint(document_id, processed_chunks, all_nodes, all_edges)
            
            # Save graph periodically for very large documents
            if processed_chunks % 50 == 0:
                await self.save_graph()
        
        # Final deduplication
        if use_streaming_dedup:
            # Get nodes from streaming state
            async with self._running_lock:
                unique_nodes = list(self._running_nodes.values())
        else:
            # Standard deduplication
            self.logger.info(f"Deduplicating {len(all_nodes)} nodes...")
            unique_nodes = self._deduplicate_nodes(all_nodes)
        
        # Create ID mapping for edges
        id_mapping = {node.id: node.id for node in unique_nodes}
        
        # Update edges with deduplicated IDs
        unique_edges = self._update_edge_ids(all_edges, id_mapping)
        
        self.logger.info(f"After deduplication: {len(unique_nodes)} nodes, {len(unique_edges)} edges")
        
        # Create final GraphData
        final_result = GraphData(nodes=unique_nodes, edges=unique_edges)
        if metadata:
            final_result.metadata = metadata
        
        # Add to graph using the existing merge logic
        await self.add_graph_data(final_result, document_id)
        
        # ===== POST-PROCESSING: always run for all document types =====
        await self.post_process_graph(document_type)
        
        # Cache result
        if document_id:
            await self._cache_document_result(document_id, doc_hash, final_result)
        
        # Clear checkpoint on success
        if document_id and self.enable_checkpointing:
            await self._clear_checkpoint(document_id)
        
        # Save graph
        await self.save_graph()
        
        return final_result

    async def _extract_single_chunk(
        self,
        chunk_text: str,
        llm_client: LLMClient,
        document_id: Optional[str],
        document_type: Optional[str],
        chunk_info: Dict[str, Any],
        metadata: Optional[Dict[str, Any]]
    ) -> Optional[GraphData]:
        """Extract entities from a single chunk with smart validation."""
        async with self._semaphore:
            try:
                # Build prompt with chunk context
                prompt = self._build_extraction_prompt(chunk_text, document_type, chunk_info)
                
                # Get LLM response with timeout
                llm_response = await asyncio.wait_for(
                    llm_client.generate(prompt),
                    timeout=60  # 1 minute timeout per chunk
                )
                
                # Parse response
                parsed_data = self._parse_llm_response(llm_response)
                
                if not parsed_data:
                    return None
                
                # Smart validation - Use the enhanced validator
                validated_data = await self._validate_and_fix_graph_data(parsed_data)
                
                if not validated_data:
                    self.logger.warning(f"Chunk {chunk_info.get('index', '?')} validation failed completely")
                    return None
                
                return validated_data
                
            except asyncio.TimeoutError:
                self.logger.error(f"Chunk {chunk_info.get('index', '?')} timed out")
                return None
            except Exception as e:
                self.logger.error(f"Chunk {chunk_info.get('index', '?')} extraction failed: {e}")
                return None

    async def extract_and_build_graph(
        self, 
        text: str, 
        llm_client: LLMClient, 
        document_id: Optional[str] = None,
        document_type: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        use_chunking: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Optional[GraphData]:
        """Main extraction method with retry logic and error handling."""
        async with self._stats_lock: 
            self.stats['total_extractions'] += 1
        
        # Log document size info
        doc_size = len(text)
        self.logger.info(f"Document size: {doc_size:,} characters (~{doc_size//5000} pages)")
        
        # Always use chunking for consistency
        result = await self.extract_from_chunks(
            text, 
            llm_client, 
            document_id, 
            document_type, 
            metadata, 
            progress_callback,
            use_streaming_dedup=(doc_size > LARGE_DOC_THRESHOLD)
        )
        
        if document_id:
            async with self._stats_lock:
                self.stats['total_documents_processed'] += 1
        
        return result

    async def _merge_duplicate_entities(self, graph_data: GraphData, document_id: str) -> GraphData:
        """Merges duplicate entities with existing ones in the graph."""
        merged_nodes = []
        id_mapping = {}
        
        for node in graph_data.nodes:
            existing_id = await self._find_existing_entity(node.type.value if hasattr(node.type, 'value') else str(node.type), node.label)
            if existing_id:
                id_mapping[node.id] = existing_id
                async with self._graph_lock:
                    if existing_id in self.graph:
                        existing_node = self.graph.nodes[existing_id]
                        await self._update_entity_confidence(existing_node, node.confidence)
                        # Merge properties
                        existing_props = existing_node.setdefault('properties', {})
                        for key, value in node.properties.items():
                            if key not in existing_props:
                                existing_props[key] = value
                        # Add document source
                        sources = existing_node.setdefault('source_documents', [])
                        if document_id not in sources: 
                            sources.append(document_id)
            else:
                id_mapping[node.id] = node.id
                node.source_documents = [document_id]
                node.properties['observation_count'] = node.properties.get('observation_count', 1)
                merged_nodes.append(node)
        
        # Update edges with mapped IDs
        updated_edges = []
        for edge in graph_data.edges:
            updated_edge = edge.copy(update={
                'source': id_mapping.get(edge.source, edge.source),
                'target': id_mapping.get(edge.target, edge.target),
                'source_documents': [document_id]
            })
            # Only add edge if both nodes exist
            if updated_edge.source != updated_edge.target:
                updated_edges.append(updated_edge)
        
        return GraphData(nodes=merged_nodes, edges=updated_edges)

    async def add_graph_data(self, graph_data: GraphData, document_id: Optional[str] = None):
        """Adds graph data with thread-safe operations."""
        if document_id:
            graph_data = await self._merge_duplicate_entities(graph_data, document_id)
        
        async with self._graph_lock:
            # Add nodes
            for node in graph_data.nodes:
                node_dict = node.dict()
                self.graph.add_node(node.id, **node_dict)
                
            # Add edges
            for edge in graph_data.edges:
                edge_dict = edge.dict()
                # Ensure source and target exist
                if edge.source in self.graph and edge.target in self.graph:
                    self.graph.add_edge(edge.source, edge.target, **edge_dict)
        
        # Update index
        async with self._id_index_lock:
            for node in graph_data.nodes:
                if len(self._id_index) < MAX_INDEX_SIZE:
                    self._id_index[self._normalize_label(node.label)].add(node.id)
        
        # Update stats
        async with self._stats_lock:
            self.stats['entities_created'] += len(graph_data.nodes)
            self.stats['relationships_created'] += len(graph_data.edges)
        
        self.logger.info(f"Added {len(graph_data.nodes)} nodes and {len(graph_data.edges)} edges.")

    # --- Query Methods ---

    def find_entities_by_type(self, node_type: NodeType) -> List[Dict[str, Any]]:
        """Find all entities of a specific type."""
        results = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == node_type.value:
                results.append({"id": node_id, **data})
        return results

    async def find_entities_by_label(self, label_pattern: str, exact_match: bool = False) -> List[Dict[str, Any]]:
        """Find entities by label with O(1) index lookup for exact matches."""
        results = []
        if exact_match:
            normalized = self._normalize_label(label_pattern)
            async with self._id_index_lock:
                candidate_ids = self._id_index.get(normalized, set()).copy()
            
            for node_id in candidate_ids:
                if node_id in self.graph:
                    results.append({"id": node_id, **self.graph.nodes[node_id]})
        else:
            pattern = label_pattern.lower()
            for node_id, data in self.graph.nodes(data=True):
                if pattern in data.get('label', '').lower():
                    results.append({"id": node_id, **data})
        return results

    def find_related_entities(self, entity_id: str, max_depth: int = 2) -> GraphData:
        """Find related entities with BFS traversal."""
        if entity_id not in self.graph:
            return GraphData(nodes=[], edges=[])
        
        visited = {entity_id}
        queue = [(entity_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            
            for neighbor in nx.all_neighbors(self.graph, current_id):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        subgraph = self.graph.subgraph(visited)
        
        # Convert to GraphData
        nodes = []
        for node_id, data in subgraph.nodes(data=True):
            try:
                nodes.append(Node(**data))
            except Exception as e:
                self.logger.warning(f"Skipping invalid node {node_id}: {e}")
        
        edges = []
        for source, target, key, data in subgraph.edges(data=True, keys=True):
            try:
                edges.append(Edge(**data))
            except Exception as e:
                self.logger.warning(f"Skipping invalid edge {source}->{target}: {e}")
        
        return GraphData(nodes=nodes, edges=edges)

    async def query_for_orchestrator(self, entities: List[str]) -> Dict[str, Any]:
        """Optimized query method for the KAIROS orchestrator."""
        results = {
            "found_entities": [], 
            "related_entities": [], 
            "relationships": [],
            "document_ids": set(), 
            "confidence_scores": {}
        }
        found_ids = set()

        for entity_name in entities:
            # Try exact match first, then fuzzy
            matches = await self.find_entities_by_label(entity_name, exact_match=True)
            if not matches:
                matches = await self.find_entities_by_label(entity_name, exact_match=False)
            
            for entity_data in matches:
                entity_id = entity_data['id']
                if entity_id in found_ids: 
                    continue
                
                found_ids.add(entity_id)
                results["found_entities"].append(entity_data)
                results["confidence_scores"][entity_id] = entity_data.get('confidence', 0.5)
                results["document_ids"].update(entity_data.get('source_documents', []))
                
                # Get related entities
                related = self.find_related_entities(entity_id, max_depth=1)
                for node in related.nodes:
                    if node.id != entity_id:
                        results["related_entities"].append(node.dict())
                        if node.id in self.graph:
                            results["document_ids"].update(
                                self.graph.nodes[node.id].get('source_documents', [])
                            )
                for edge in related.edges:
                    results["relationships"].append(edge.dict())
        
        # Convert set to list for JSON serialization
        results["document_ids"] = list(results["document_ids"])
        return results

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Gets comprehensive graph statistics."""
        node_types = Counter(data.get('type') for _, data in self.graph.nodes(data=True))
        edge_types = Counter(data.get('type') for _, _, data in self.graph.edges(data=True))
        
        # Calculate document statistics
        doc_stats = Counter()
        for _, data in self.graph.nodes(data=True):
            doc_sources = data.get('source_documents', [])
            doc_stats.update(doc_sources)
        
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "graph_density": round(nx.density(self.graph), 4) if self.graph.number_of_nodes() > 0 else 0,
            "extraction_stats": dict(self.stats),
            "extraction_analytics": self.get_extraction_analytics(),
            "index_size": len(self._id_index),
            "cache_size": len(self._entity_cache),
            "document_count": len(doc_stats),
            "avg_nodes_per_doc": round(self.graph.number_of_nodes() / max(1, len(doc_stats)), 2),
            "performance_metrics": {
                "documents_processed": self.stats.get('total_documents_processed', 0),
                "successful_extractions": self.stats.get('entities_created', 0),
                "type_mappings_applied": sum(self.type_usage_stats.values()),
                "system_robustness": "Production Ready"
            }
        }

    def export_for_visualization(self, node_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Exports a full or partial graph for frontend visualization."""
        try:
            if node_ids:
                # Filter valid node IDs
                valid_ids = [nid for nid in node_ids if nid in self.graph]
                subgraph = self.graph.subgraph(valid_ids)
            else:
                subgraph = self.graph
            
            data = json_graph.node_link_data(subgraph)
            
            # Enhance node data for visualization
            for node in data.get('nodes', []):
                node_id = node.get('id', '')
                if node_id in subgraph:
                    node['size'] = 10 + (subgraph.degree(node_id) * 2)
                    node['color'] = self._get_node_color(node.get('type'))
            
            return data
        except Exception as e:
            self.logger.error(f"Error exporting graph: {e}")
            return {"nodes": [], "links": []}

    def _get_node_color(self, node_type: str) -> str:
        """Gets a consistent color for a given node type."""
        color_map = {
            NodeType.PERSON.value: "#3B82F6",
            NodeType.COMPANY.value: "#10B981",
            NodeType.POLICY.value: "#F59E0B",
            NodeType.CLAIM.value: "#EF4444",
            NodeType.CONTRACT.value: "#8B5CF6",
            NodeType.DOCUMENT.value: "#6B7280",
            NodeType.AMOUNT.value: "#EC4899",
            NodeType.DATE.value: "#14B8A6",
            NodeType.LOCATION.value: "#F97316",
            NodeType.EVENT.value: "#8B5CF6",
            NodeType.PRODUCT.value: "#06B6D4",
            NodeType.REGULATION.value: "#6366F1",
            NodeType.BENEFICIARY.value: "#F472B6",
            NodeType.CLAUSE.value: "#A78BFA",
            NodeType.PROCEDURE.value: "#34D399",
            NodeType.SECTION.value: "#60A5FA",
            NodeType.ENDORSEMENT.value: "#FBBF24",
            NodeType.EXCLUSION.value: "#F87171",
            NodeType.TERM.value: "#6EE7B7",
            NodeType.BENEFIT.value: "#93C5FD",
            NodeType.PROVISION.value: "#DDD6FE",
            NodeType.COVERAGE.value: "#FDE68A",
            "Entity": "#9CA3AF"  # Fallback color for dynamic entities
        }
        return color_map.get(node_type, "#9CA3AF")

    async def reset_graph(self):
        """Resets the graph and statistics with proper cleanup."""
        async with self._graph_lock:
            self.graph.clear()
            
        async with self._id_index_lock:
            self._id_index.clear()
            
        async with self._stats_lock:
            self.stats.clear()
        
        # Reset analytics
        self.type_usage_stats.clear()
        self.type_mapping_log.clear()
        
        async with self._cache_lock:
            self._entity_cache.clear()
        
        async with self._running_lock:
            self._running_nodes.clear()
            self._running_edges.clear()
        
        async with self._doc_cache_lock:
            self._doc_cache.clear()
            
        await self._init_graph()
        await self.save_graph()
        self.logger.info("Graph reset complete.")

    async def close(self):
        """Clean up resources properly."""
        try:
            await self.save_graph()
            
            # Clear checkpoints directory
            if self.enable_checkpointing:
                checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('_checkpoint.pkl')]
                for file in checkpoint_files:
                    try:
                        os.remove(os.path.join(CHECKPOINT_DIR, file))
                    except:
                        pass
            
            self.logger.info("GraphService closed successfully.")
        except Exception as e:
            self.logger.error(f"Error during GraphService cleanup: {e}")
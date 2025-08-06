# backend/app/models/schemas.py

from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict, Any, Set, Union
from enum import Enum
from datetime import datetime
from collections import Counter
import uuid
import re

# 🧠 SMART NODE TYPE VALIDATOR FOR HACKRX 6.0 - THE CORE FIX
class SmartNodeTypeValidator:
    """
    🏆 HACKRX 6.0 SOLUTION: Intelligent node type validator that maps ANY LLM output to valid types
    This solves the core validation errors from your logs: Form -> Document, Property -> Entity, etc.
    """
    
    TYPE_MAPPINGS = {
        # Document-related (handles "Form", "Application", etc.)
        'form': 'Document', 'application': 'Document', 'certificate': 'Document',
        'statement': 'Document', 'report': 'Document', 'letter': 'Document',
        'contract': 'Document', 'agreement': 'Document', 'policy': 'Policy',
        'manual': 'Document', 'guide': 'Document', 'specification': 'Document',
        'receipt': 'Document', 'invoice': 'Document', 'bill': 'Document',
        
        # Property/Asset-related (handles "Property", "Asset", etc.)
        'property': 'Entity', 'asset': 'Entity', 'item': 'Entity',
        'object': 'Entity', 'resource': 'Entity', 'component': 'Entity',
        'feature': 'Entity', 'attribute': 'Entity', 'characteristic': 'Entity',
        'element': 'Entity', 'factor': 'Entity', 'aspect': 'Entity',
        
        # Action-related (handles "Action", "Activity", etc.)
        'action': 'Procedure', 'activity': 'Procedure', 'process': 'Procedure',
        'step': 'Procedure', 'task': 'Procedure', 'operation': 'Procedure',
        'method': 'Procedure', 'workflow': 'Procedure', 'protocol': 'Procedure',
        'treatment': 'Procedure', 'therapy': 'Procedure', 'surgery': 'Procedure',
        
        # Value-related
        'value': 'Amount', 'cost': 'Amount', 'price': 'Amount',
        'fee': 'Amount', 'charge': 'Amount', 'sum': 'Amount', 'limit': 'Amount',
        'budget': 'Amount', 'expense': 'Amount', 'revenue': 'Amount',
        'salary': 'Amount', 'wage': 'Amount', 'payment': 'Amount',
        
        # Time-related
        'time': 'Period', 'duration': 'Period', 'interval': 'Period',
        'deadline': 'Date', 'schedule': 'Period', 'timeline': 'Period',
        'appointment': 'Date', 'meeting': 'Event', 'session': 'Event',
        
        # Status-related
        'status': 'Condition', 'state': 'Condition', 'situation': 'Condition',
        'outcome': 'Condition', 'result': 'Condition', 'response': 'Condition',
        'diagnosis': 'Condition', 'symptom': 'Condition', 'side_effect': 'Condition',
        
        # Medical/Health specific (for medical documents)
        'medication': 'Entity', 'drug': 'Entity', 'medicine': 'Entity',
        'hospital': 'Location', 'clinic': 'Location', 'doctor': 'Person',
        'patient': 'Person', 'physician': 'Person', 'nurse': 'Person',
        
        # Legal specific (for legal documents)
        'obligation': 'Requirement', 'right': 'Benefit', 'duty': 'Requirement',
        'clause': 'Clause', 'section': 'Section', 'article': 'Section',
        'law': 'Regulation', 'rule': 'Term', 'regulation': 'Regulation',
        
        # Insurance specific (for insurance documents)
        'rider': 'Provision', 'endorsement': 'Provision', 'exclusion': 'Exclusion',
        'coverage': 'Coverage', 'deductible': 'Amount', 'premium': 'Amount',
        'copay': 'Amount', 'coinsurance': 'Amount', 'beneficiary': 'Beneficiary',
        
        # Business/Financial specific
        'organization': 'Company', 'corporation': 'Company', 'firm': 'Company',
        'enterprise': 'Company', 'business': 'Company', 'vendor': 'Company',
        'supplier': 'Company', 'client': 'Company', 'customer': 'Person',
        'account': 'Entity', 'transaction': 'Event', 'investment': 'Entity',
        
        # Technical specific (for technical documents)
        'system': 'Entity', 'software': 'Entity', 'hardware': 'Entity',
        'database': 'Entity', 'server': 'Entity', 'api': 'Entity',
        'interface': 'Entity', 'module': 'Entity',
        
        # Location specific
        'address': 'Location', 'office': 'Location', 'building': 'Location',
        'facility': 'Location', 'site': 'Location', 'venue': 'Location',
        'country': 'Location', 'city': 'Location', 'state': 'Location',
    }
    
    @classmethod
    def normalize_type(cls, node_type: str) -> str:
        """🎯 Convert ANY node type to a valid NodeType - THE CORE FIX"""
        if not node_type:
            return "Entity"
        
        # Check if already valid (avoid unnecessary processing)
        try:
            # Import here to avoid circular import
            from . import NodeType
            NodeType(node_type)
            return node_type
        except (ValueError, ImportError):
            pass
        
        # Normalize input for mapping
        normalized = node_type.lower().strip()
        normalized = re.sub(r'^(the|a|an)\s+', '', normalized)  # Remove articles
        normalized = re.sub(r'[^a-z0-9_]+', '_', normalized).strip('_')  # Clean special chars
        
        # Direct mapping (most common cases)
        if normalized in cls.TYPE_MAPPINGS:
            return cls.TYPE_MAPPINGS[normalized]
        
        # Partial matching for compound terms
        for pattern, mapped_type in cls.TYPE_MAPPINGS.items():
            if pattern in normalized or normalized in pattern:
                return mapped_type
        
        # 🧠 Semantic analysis (AI-like intelligence)
        if any(word in normalized for word in ['money', 'dollar', 'cost', 'pay', 'price', 'fee', 'charge']):
            return "Amount"
        elif any(word in normalized for word in ['date', 'year', 'month', 'day', 'time', 'when']):
            return "Date"
        elif any(word in normalized for word in ['person', 'individual', 'customer', 'client', 'user', 'member']):
            return "Person"
        elif any(word in normalized for word in ['company', 'corporation', 'organization', 'business', 'firm']):
            return "Company"
        elif any(word in normalized for word in ['rule', 'condition', 'term', 'requirement', 'clause']):
            return "Term"
        elif any(word in normalized for word in ['place', 'location', 'address', 'site', 'facility']):
            return "Location"
        elif any(word in normalized for word in ['coverage', 'benefit', 'protection', 'insurance']):
            return "Coverage"
        elif any(word in normalized for word in ['procedure', 'process', 'action', 'step', 'method']):
            return "Procedure"
        elif any(word in normalized for word in ['document', 'form', 'report', 'statement', 'certificate']):
            return "Document"
        elif any(word in normalized for word in ['event', 'occurrence', 'incident', 'happening']):
            return "Event"
        
        # 🛡️ Safe fallback - NEVER fails
        return "Entity"

# --- ENUMs for controlled vocabularies ---

class NodeType(str, Enum):
    PERSON = "Person"
    POLICY = "Policy"
    COMPANY = "Company"
    DOCUMENT = "Document"
    CLAIM = "Claim"
    CONTRACT = "Contract"
    AMOUNT = "Amount"
    DATE = "Date"
    LOCATION = "Location"
    EVENT = "Event"
    PRODUCT = "Product"
    REGULATION = "Regulation"
    DEDUCTIBLE = "Deductible"
    BENEFICIARY = "Beneficiary"
    CLAUSE = "Clause"
    PROCEDURE = "Procedure"
    SECTION = "Section"
    ENDORSEMENT = "Endorsement"
    EXCLUSION = "Exclusion"
    # HackRx 6.0 Enhanced Types
    TERM = "Term"
    BENEFIT = "Benefit"
    PROVISION = "Provision"
    COVERAGE = "Coverage"
    ENTITY = "Entity"  # 🏆 Fallback type for unknown entities
    CONDITION = "Condition"  # For medical/status conditions
    PERIOD = "Period"  # For time periods
    REQUIREMENT = "Requirement"  # For obligations/requirements

class RelationshipType(str, Enum):
    """Common relationship types in the knowledge graph."""
    # Ownership & Control
    OWNS = "owns"
    OWNED_BY = "owned_by"
    CONTROLS = "controls"
    CONTROLLED_BY = "controlled_by"
    
    # Employment & Roles
    WORKS_FOR = "works_for"
    EMPLOYS = "employs"
    DIRECTOR_OF = "director_of"
    MEMBER_OF = "member_of"
    
    # Document Relations
    SIGNED = "signed"
    AUTHORED = "authored"
    MENTIONED_IN = "mentioned_in"
    REFERENCES = "references"
    
    # Legal & Compliance
    GOVERNS = "governs"
    GOVERNED_BY = "governed_by"
    VIOLATES = "violates"
    COMPLIES_WITH = "complies_with"
    
    # Temporal
    PRECEDED_BY = "preceded_by"
    FOLLOWED_BY = "followed_by"
    CONCURRENT_WITH = "concurrent_with"
    
    # Generic
    RELATED_TO = "related_to"
    ASSOCIATED_WITH = "associated_with"

class EntityConfidence(str, Enum):
    """Categorical confidence levels for easier filtering and UI representation."""
    HIGH = "high"      # > 0.8
    MEDIUM = "medium"  # 0.5 - 0.8
    LOW = "low"        # < 0.5

class ProcessingStatus(str, Enum):
    """Status of document processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # Some extraction succeeded, some failed
    CACHED = "cached"    # Document already processed and cached

class DocumentType(str, Enum):
    """Types of documents that can be processed."""
    CONTRACT = "contract"
    POLICY = "policy"
    REPORT = "report"
    EMAIL = "email"
    MEMO = "memo"
    LEGAL = "legal"
    FINANCIAL = "financial"
    OTHER = "other"

# --- Base Models with Common Fields ---

class TimestampedModel(BaseModel):
    """Base model with timestamp fields for auditability."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class VersionedModel(BaseModel):
    """Base model with version tracking for managing changes."""
    version: int = Field(default=1, ge=1, description="Version number for tracking changes")
    
    def increment_version(self):
        self.version += 1
        if hasattr(self, 'updated_at'):
            self.updated_at = datetime.utcnow()

# --- Graph Data Models ---

class Node(TimestampedModel, VersionedModel):
    """🏆 HackRx 6.0 Enhanced Node with Smart Type Validation - Represents a single entity in the knowledge graph."""
    id: str = Field(default_factory=lambda: f"node_{uuid.uuid4()}", description="Auto-generated unique node ID.")
    type: Union[NodeType, str] = Field(..., description="Node type - automatically normalized to valid enum")  # 🎯 Allow both enum and string
    label: str = Field(..., min_length=1, max_length=200, description="A human-readable label for the node.")
    properties: Dict[str, Any] = Field(default_factory=dict, description="A dictionary of attributes for the node.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="The confidence score of the entity extraction.")
    source_documents: List[str] = Field(default_factory=list, description="IDs of documents from which this node was extracted.")
    aliases: List[str] = Field(default_factory=list, description="Alternative names or identifiers for this entity.")
    tags: Set[str] = Field(default_factory=set, description="Tags for categorization and filtering.")
    
    # 🏆 HACKRX 6.0 SMART VALIDATORS - THE MAIN FIX
    @validator('type', pre=True)
    def normalize_node_type(cls, v):
        """🎯 Automatically normalize any node type to valid enum - CORE FIX"""
        if isinstance(v, str):
            normalized = SmartNodeTypeValidator.normalize_type(v)
            return normalized
        return v
    
    @validator('type')
    def validate_type_is_enum(cls, v):
        """🛡️ Ensure the type is a valid NodeType enum value with fallback"""
        if isinstance(v, str):
            try:
                return NodeType(v)
            except ValueError:
                # Final fallback to Entity if somehow validation failed
                return NodeType.ENTITY
        return v
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Rounds the confidence score to two decimal places."""
        return round(v, 2)
    
    @validator('label')
    def clean_label(cls, v):
        """Cleans and normalizes the label."""
        return re.sub(r'\s+', ' ', v.strip())
    
    @validator('aliases')
    def clean_aliases(cls, v):
        """Removes duplicates and empty aliases."""
        return list(set(alias.strip() for alias in v if alias.strip()))
    
    @validator('properties')
    def convert_sets_to_lists(cls, v):
        """Convert any sets in properties to lists for JSON serialization."""
        if isinstance(v, dict):
            for key, value in v.items():
                if isinstance(value, set):
                    v[key] = list(value)
                elif isinstance(value, dict):
                    for k, val in value.items():
                        if isinstance(val, set):
                            value[k] = list(val)
        return v
    
    @validator('tags', pre=True)
    def convert_tags_set_to_list(cls, v):
        """Convert tags set to list for JSON serialization."""
        if isinstance(v, set):
            return list(v)
        return v
    
    @validator('source_documents')
    def convert_source_documents_set_to_list(cls, v):
        """Convert source_documents set to list for JSON serialization."""
        if isinstance(v, set):
            return list(v)
        return v
    
    @property
    def confidence_level(self) -> EntityConfidence:
        """Computes a categorical confidence level based on the numerical score."""
        if self.confidence > 0.8:
            return EntityConfidence.HIGH
        elif self.confidence >= 0.5:
            return EntityConfidence.MEDIUM
        return EntityConfidence.LOW
    
    @property
    def display_name(self) -> str:
        """Returns the best display name for the entity."""
        return self.label
    
    def merge_with(self, other: 'Node') -> 'Node':
        """Merges this node with another node, combining their properties."""
        if self.type != other.type:
            raise ValueError(f"Cannot merge nodes of different types: {self.type} and {other.type}")
        
        merged_props = {**self.properties, **other.properties}
        merged_sources = list(set(self.source_documents + other.source_documents))
        merged_aliases = list(set(self.aliases + other.aliases + [self.label, other.label]))
        merged_confidence = max(self.confidence, other.confidence)
        
        # Use the label with higher confidence
        label = self.label if self.confidence >= other.confidence else other.label
        
        return Node(
            type=self.type,
            label=label,
            properties=merged_props,
            confidence=merged_confidence,
            source_documents=merged_sources,
            aliases=[alias for alias in merged_aliases if alias != label],  # Remove chosen label from aliases
            tags=self.tags.union(other.tags) if isinstance(self.tags, set) else set(self.tags).union(set(other.tags)),
            version=max(self.version, other.version) + 1
        )

class Edge(TimestampedModel):
    """Represents a relationship (an edge) between two nodes."""
    id: str = Field(default_factory=lambda: f"edge_{uuid.uuid4()}", description="Auto-generated unique edge ID.")
    source: str = Field(..., description="The ID of the source node.")
    target: str = Field(..., description="The ID of the target node.")
    type: Union[RelationshipType, str] = Field(..., description="The type of relationship.")
    label: str = Field(..., min_length=1, max_length=100, description="A description of the relationship.")
    properties: Dict[str, Any] = Field(default_factory=dict, description="A dictionary of attributes for the edge.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="The confidence score of the extracted relationship.")
    bidirectional: bool = Field(default=False, description="Indicates if the relationship is bidirectional.")
    weight: float = Field(default=1.0, ge=0.0, description="Weight of the relationship for graph algorithms.")
    source_documents: List[str] = Field(default_factory=list, description="IDs of documents supporting this relationship.")
    
    @validator('type')
    def validate_relationship_type(cls, v):
        """Allows both enum values and custom strings for flexibility."""
        if isinstance(v, str) and not isinstance(v, RelationshipType):
            try:
                return RelationshipType(v.lower().replace(' ', '_'))
            except ValueError:
                return v
        return v
    
    @validator('properties')
    def convert_sets_to_lists(cls, v):
        """Convert any sets in properties to lists for JSON serialization."""
        if isinstance(v, dict):
            for key, value in v.items():
                if isinstance(value, set):
                    v[key] = list(value)
                elif isinstance(value, dict):
                    for k, val in value.items():
                        if isinstance(val, set):
                            value[k] = list(val)
        return v
    
    @validator('source_documents')
    def convert_source_documents_set_to_list(cls, v):
        """Convert source_documents set to list for JSON serialization."""
        if isinstance(v, set):
            return list(v)
        return v
    
    def reverse(self) -> 'Edge':
        """Creates a reversed version of this edge."""
        reverse_map = {
            RelationshipType.OWNS: RelationshipType.OWNED_BY,
            RelationshipType.OWNED_BY: RelationshipType.OWNS,
            RelationshipType.CONTROLS: RelationshipType.CONTROLLED_BY,
            RelationshipType.CONTROLLED_BY: RelationshipType.CONTROLS,
            RelationshipType.EMPLOYS: RelationshipType.WORKS_FOR,
            RelationshipType.WORKS_FOR: RelationshipType.EMPLOYS,
            RelationshipType.PRECEDED_BY: RelationshipType.FOLLOWED_BY,
            RelationshipType.FOLLOWED_BY: RelationshipType.PRECEDED_BY,
        }
        reverse_type = reverse_map.get(self.type, f"reverse_{self.type}") if isinstance(self.type, RelationshipType) else f"reverse_{self.type}"
        
        return Edge(
            source=self.target,
            target=self.source,
            type=reverse_type,
            label=f"reverse of {self.label}",
            properties=self.properties,
            confidence=self.confidence,
            weight=self.weight,
            source_documents=self.source_documents
        )

class GraphData(BaseModel):
    """Represents the entire graph structure extracted from a document."""
    nodes: List[Node]
    edges: List[Edge]
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Graph-level metadata.")
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('nodes')
    def validate_unique_nodes(cls, v):
        """Ensures that all node IDs within a single graph extraction are unique."""
        node_ids = [node.id for node in v]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError('All node IDs must be unique')
        return v
    
    @validator('metadata')
    def convert_metadata_sets_to_lists(cls, v):
        """Convert any sets in metadata to lists for JSON serialization."""
        if isinstance(v, dict):
            for key, value in v.items():
                if isinstance(value, set):
                    v[key] = list(value)
                elif isinstance(value, dict):
                    for k, val in value.items():
                        if isinstance(val, set):
                            value[k] = list(val)
        return v
    
    @root_validator(skip_on_failure=True)
    def validate_edge_references(cls, values):
        """Ensures that every edge connects to valid nodes within the same graph data payload."""
        nodes, edges = values.get('nodes', []), values.get('edges', [])
        node_ids = {node.id for node in nodes}
        for edge in edges:
            if edge.source not in node_ids:
                raise ValueError(f'Edge source "{edge.source}" not found in nodes')
            if edge.target not in node_ids:
                raise ValueError(f'Edge target "{edge.target}" not found in nodes')
        return values
    
    @property
    def is_empty(self) -> bool:
        """Check if the graph is empty."""
        return len(self.nodes) == 0 and len(self.edges) == 0
    
    @property
    def statistics(self) -> Dict[str, Any]:
        """Computes statistics about the graph."""
        if not self.nodes: return {}
        node_type_counts = Counter(node.type for node in self.nodes)
        edge_type_counts = Counter(str(edge.type) for edge in self.edges)
        avg_node_confidence = sum(n.confidence for n in self.nodes) / len(self.nodes)
        avg_edge_confidence = sum(e.confidence for e in self.edges) / len(self.edges) if self.edges else 0
        
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_types": dict(node_type_counts),
            "edge_types": dict(edge_type_counts),
            "avg_node_confidence": round(avg_node_confidence, 2),
            "avg_edge_confidence": round(avg_edge_confidence, 2),
            "density": self._calculate_density(),
            "connected_components": self._count_components(),
        }
    
    def _calculate_density(self) -> float:
        """Calculates graph density (ratio of actual edges to possible edges)."""
        n = len(self.nodes)
        if n <= 1:
            return 0.0
        max_edges = n * (n - 1)  # For directed graph
        return round(len(self.edges) / max_edges, 4) if max_edges > 0 else 0.0
    
    def _count_components(self) -> int:
        """Counts the number of connected components in the graph."""
        if not self.nodes:
            return 0
        
        adj = {node.id: set() for node in self.nodes}
        for edge in self.edges:
            adj[edge.source].add(edge.target)
            adj[edge.target].add(edge.source) # Treat as undirected for component counting
            
        visited = set()
        components = 0
        
        def dfs(node_id):
            visited.add(node_id)
            for neighbor in adj[node_id]:
                if neighbor not in visited:
                    dfs(neighbor)
        
        for node in self.nodes:
            if node.id not in visited:
                dfs(node.id)
                components += 1
        
        return components
    
    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Retrieves a node by its ID."""
        return next((node for node in self.nodes if node.id == node_id), None)
    
    def get_node_neighbors(self, node_id: str) -> List[Node]:
        """Gets all neighboring nodes of a given node."""
        neighbor_ids = set()
        for edge in self.edges:
            if edge.source == node_id:
                neighbor_ids.add(edge.target)
            if edge.target == node_id:
                neighbor_ids.add(edge.source)
        
        return [node for node in self.nodes if node.id in neighbor_ids]
    
    def filter_by_confidence(self, min_confidence: float) -> 'GraphData':
        """Returns a new GraphData with only high-confidence nodes and edges."""
        filtered_nodes = [n for n in self.nodes if n.confidence >= min_confidence]
        filtered_node_ids = {n.id for n in filtered_nodes}
        
        filtered_edges = [
            e for e in self.edges 
            if e.confidence >= min_confidence 
            and e.source in filtered_node_ids 
            and e.target in filtered_node_ids
        ]
        
        return GraphData(
            nodes=filtered_nodes,
            edges=filtered_edges,
            metadata={**self.metadata, "filtered_confidence": min_confidence}
        )

# --- API Request/Response & Helper Models ---

class DocumentMetadata(TimestampedModel):
    """Model for tracking metadata about an uploaded document."""
    document_id: str = Field(default_factory=lambda: f"doc_{uuid.uuid4()}")
    filename: str
    file_size: int = Field(..., gt=0, description="File size in bytes")
    mime_type: str
    checksum: Optional[str] = Field(None, description="SHA-256 checksum of the file")
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    document_type: Optional[DocumentType] = None
    language: str = Field(default="en", pattern="^[a-z]{2}$")
    page_count: Optional[int] = Field(None, ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('mime_type')
    def validate_mime_type(cls, v):
        """Validates that the MIME type is supported."""
        supported_types = {
            'application/pdf',
            'text/plain',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'text/html',
            'application/json'
        }
        if v not in supported_types:
            raise ValueError(f'Unsupported MIME type: {v}')
        return v
    
    @validator('metadata')
    def convert_metadata_sets_to_lists(cls, v):
        """Convert any sets in metadata to lists for JSON serialization."""
        if isinstance(v, dict):
            for key, value in v.items():
                if isinstance(value, set):
                    v[key] = list(value)
                elif isinstance(value, dict):
                    for k, val in value.items():
                        if isinstance(val, set):
                            value[k] = list(val)
        return v

class ProcessingResult(BaseModel):
    """A structured model for returning the result of an asynchronous processing job."""
    document_id: str
    status: ProcessingStatus
    chunks_created: int = Field(default=0, ge=0)
    entities_extracted: int = Field(default=0, ge=0)
    relationships_found: int = Field(default=0, ge=0)
    processing_time: float = Field(default=0.0, ge=0.0, description="Processing time in seconds")
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    graph_data: Optional[GraphData] = None
    processing_stages: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Detailed timing and results for each processing stage"
    )
    
    @validator('processing_stages')
    def convert_processing_stages_sets_to_lists(cls, v):
        """Convert any sets in processing_stages to lists for JSON serialization."""
        if isinstance(v, dict):
            for key, value in v.items():
                if isinstance(value, dict):
                    for k, val in value.items():
                        if isinstance(val, set):
                            value[k] = list(val)
                elif isinstance(value, set):
                    v[key] = list(value)
        return v
    
    @property
    def success_rate(self) -> float:
        """Calculates the success rate of extraction."""
        if self.status == ProcessingStatus.SUCCESS:
            return 1.0
        elif self.status == ProcessingStatus.PARTIAL:
            return 0.5  # Simplified - could be more sophisticated
        return 0.0

class QueryRequest(BaseModel):
    """Defines the expected structure for a user's query to the chat endpoint."""
    query: str = Field(..., min_length=1, max_length=1000)
    document_ids: Optional[List[str]] = Field(None, description="Filter results to specific documents")
    node_types: Optional[List[NodeType]] = Field(None, description="Filter by node types")
    include_graph: bool = Field(default=True, description="Include graph visualization data")
    max_results: int = Field(default=5, ge=1, le=50)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    search_mode: str = Field(default="hybrid", pattern="^(vector|keyword|hybrid)$")
    include_reasoning: bool = Field(default=False, description="Include reasoning trace in response")
    
    @validator('query')
    def clean_query(cls, v):
        """Cleans and normalizes the query."""
        return re.sub(r'\s+', ' ', v.strip())

class GraphQueryRequest(BaseModel):
    """Simple model for querying the graph with entities."""
    entities: List[str] = Field(..., description="List of entities to search for in the graph")
    max_hops: int = Field(default=2, ge=1, le=3, description="Maximum hops from entity nodes")
    include_properties: bool = Field(default=True, description="Include node properties in response")

class SearchResult(BaseModel):
    """Individual search result from vector or keyword search."""
    content: str
    score: float
    source_type: str  # "vector" or "keyword"
    document_id: str
    chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('metadata')
    def convert_metadata_sets_to_lists(cls, v):
        """Convert any sets in metadata to lists for JSON serialization."""
        if isinstance(v, dict):
            for key, value in v.items():
                if isinstance(value, set):
                    v[key] = list(value)
                elif isinstance(value, dict):
                    for k, val in value.items():
                        if isinstance(val, set):
                            value[k] = list(val)
        return v

class QueryResponse(BaseModel):
    """Defines the structure of the response sent back to the user from the chat endpoint."""
    query_id: str = Field(default_factory=lambda: f"query_{uuid.uuid4()}")
    answer: str
    sources: List[str] = Field(..., description="List of source document IDs")
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")
    graph_context: Optional[GraphData] = None
    search_results: List[SearchResult] = Field(default_factory=list)
    reasoning_trace: Optional[List[str]] = Field(None, description="Step-by-step reasoning if requested")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    suggested_queries: List[str] = Field(default_factory=list, max_items=5)
    
    @validator('confidence')
    def round_confidence(cls, v):
        """Rounds confidence to 2 decimal places."""
        return round(v, 2)
    
    @validator('metadata')
    def convert_metadata_sets_to_lists(cls, v):
        """Convert any sets in metadata to lists for JSON serialization."""
        if isinstance(v, dict):
            for key, value in v.items():
                if isinstance(value, set):
                    v[key] = list(value)
                elif isinstance(value, dict):
                    for k, val in value.items():
                        if isinstance(val, set):
                            value[k] = list(val)
        return v
    
    @property
    def confidence_level(self) -> EntityConfidence:
        """Returns categorical confidence level."""
        if self.confidence > 0.8:
            return EntityConfidence.HIGH
        elif self.confidence >= 0.5:
            return EntityConfidence.MEDIUM
        return EntityConfidence.LOW

# --- HackRx Specific Models ---

class HackRxQueryResult(BaseModel):
    """Specific result model for HackRx queries."""
    question: str
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning: str = ""
    processing_time: float = 0.0
    
    @validator('confidence')
    def round_confidence(cls, v):
        """Rounds confidence to 2 decimal places."""
        return round(v, 2)
    
    @validator('source_chunks')
    def convert_source_chunks_sets_to_lists(cls, v):
        """Convert any sets in source_chunks to lists for JSON serialization."""
        if isinstance(v, list):
            for chunk in v:
                if isinstance(chunk, dict):
                    for key, value in chunk.items():
                        if isinstance(value, set):
                            chunk[key] = list(value)
                        elif isinstance(value, dict):
                            for k, val in value.items():
                                if isinstance(val, set):
                                    value[k] = list(val)
        return v

class HackRxProcessingMetrics(BaseModel):
    """Metrics for HackRx processing."""
    total_questions: int
    successful_answers: int
    average_confidence: float
    total_processing_time: float
    vector_search_hits: int
    graph_enrichments: int
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of answers."""
        if self.total_questions == 0:
            return 0.0
        return round(self.successful_answers / self.total_questions, 2)

# --- Utility Models ---

class HealthCheck(BaseModel):
    """Health check response model."""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    dependencies: Dict[str, bool] = Field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        """Checks if all dependencies are healthy."""
        return all(self.dependencies.values())

class TaskStatus(BaseModel):
    """Model for tracking async task status."""
    task_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Calculates task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
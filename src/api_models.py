from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ToolStatus = Literal["ok", "needs_configuration", "error"]


class FeatureCard(BaseModel):
    key: str
    title: str
    category: str
    summary: str
    interactive: bool
    loaded: bool = False


class PlatformHealthResponse(BaseModel):
    status: str
    ui: str
    streamlit_replaced: bool
    uptime_seconds: float
    python_version: str
    api_port: int
    feature_count: int
    interactive_count: int
    loaded_features: List[str]


class FeatureCatalogResponse(BaseModel):
    features: List[FeatureCard]
    categories: List[str]


class ToolResponseBase(BaseModel):
    status: ToolStatus
    tool: str
    mode: str
    message: Optional[str] = None
    required_env: List[str] = Field(default_factory=list)


class CodeMixedRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)


class DetectedEntity(BaseModel):
    type: str
    value: str
    start: Optional[int] = None
    end: Optional[int] = None


class CodeMixedResponse(ToolResponseBase):
    input_text: str
    primary_language: str
    intent: str
    confidence: float
    response_text: str
    entities: List[DetectedEntity] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)


class SentimentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class SentimentResponse(ToolResponseBase):
    label: str
    confidence: float
    threshold: float
    above_threshold: bool
    language: str
    model: str
    distribution: Dict[str, float] = Field(default_factory=dict)


class CloudToolResponse(ToolResponseBase):
    configured: bool = False
    provider: Optional[str] = None
    raw: Dict[str, Any] = Field(default_factory=dict)


class StructuredEntity(BaseModel):
    label: str
    value: str
    confidence: Optional[float] = None


class GSTINRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)
    company_name: Optional[str] = Field(default=None, max_length=200)


class GSTINResponse(CloudToolResponse):
    gstins: List[str] = Field(default_factory=list)
    company_name: Optional[str] = None
    reconciliation: Dict[str, Any] = Field(default_factory=dict)


class KYCResponse(CloudToolResponse):
    entities: List[StructuredEntity] = Field(default_factory=list)
    document_summary: Dict[str, Any] = Field(default_factory=dict)


class SLARequest(BaseModel):
    priority: str
    category: str
    customer_tier: str
    complexity: str
    created_hour: int = Field(default=10, ge=0, le=23)
    is_weekend: int = Field(default=0, ge=0, le=1)
    agent_experience: float = Field(default=2.5, ge=0.0, le=30.0)
    customer_history: int = Field(default=25, ge=0, le=1000)
    ticket_id: Optional[str] = Field(default="WEB-TICKET")


class SLAResponse(ToolResponseBase):
    ticket_id: str
    breach_probability: float
    risk_level: str
    confidence: float
    breach_prediction: int
    recommendations: List[str] = Field(default_factory=list)
    feature_contributions: Dict[str, Any] = Field(default_factory=dict)


class RagDocumentRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=20000)
    category: str = Field(min_length=1, max_length=100)
    author: str = Field(default="Frontend Operator", max_length=100)
    tags: List[str] = Field(default_factory=list)


class RagDocumentResponse(ToolResponseBase):
    document_id: str
    total_documents: int
    categories: List[str] = Field(default_factory=list)
    statistics: Dict[str, Any] = Field(default_factory=dict)


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    category: Optional[str] = Field(default=None, max_length=100)
    top_k: int = Field(default=5, ge=1, le=10)


class RagSearchResultItem(BaseModel):
    title: str
    category: str
    author: str
    similarity_score: float
    answer: str
    relevant_chunks: List[str] = Field(default_factory=list)


class RagSearchResponse(ToolResponseBase):
    results: List[RagSearchResultItem] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    statistics: Dict[str, Any] = Field(default_factory=dict)


class AuditLogRequest(BaseModel):
    action: str = Field(min_length=1, max_length=120)
    user_id: str = Field(min_length=1, max_length=80)
    resource: str = Field(min_length=1, max_length=120)
    details: Dict[str, Any] = Field(default_factory=dict)


class AuditEntryRecord(BaseModel):
    timestamp: str
    action: str
    user_id: str
    resource: str
    details: Dict[str, Any] = Field(default_factory=dict)
    hash: str
    previous_hash: str
    signature: str


class AuditVerifyResponse(ToolResponseBase):
    total_entries: int = 0
    valid_entries: int = 0
    invalid_entries: int = 0
    chain_broken: bool = False
    integrity_score: float = 0.0
    issues: List[str] = Field(default_factory=list)


class AuditLogResponse(ToolResponseBase):
    entry_id: str
    entry: Optional[AuditEntryRecord] = None
    integrity: AuditVerifyResponse


class InvoiceParseResponse(CloudToolResponse):
    file_name: Optional[str] = None
    invoice: Dict[str, Any] = Field(default_factory=dict)
    validation: Dict[str, Any] = Field(default_factory=dict)


class MeetingAnalysisResponse(CloudToolResponse):
    file_name: Optional[str] = None
    summary: Dict[str, Any] = Field(default_factory=dict)
    report: Optional[str] = None

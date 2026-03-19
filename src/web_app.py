"""
FastAPI backend for the Enterprise Control Plane.
Exposes typed platform telemetry and nine live tool workflows for the Next.js frontend.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .api_models import (
    AuditEntryRecord,
    AuditLogRequest,
    AuditLogResponse,
    AuditVerifyResponse,
    CloudToolResponse,
    CodeMixedRequest,
    CodeMixedResponse,
    DetectedEntity,
    FeatureCard,
    FeatureCatalogResponse,
    GSTINRequest,
    GSTINResponse,
    InvoiceParseResponse,
    KYCResponse,
    MeetingAnalysisResponse,
    PlatformHealthResponse,
    RagDocumentRequest,
    RagDocumentResponse,
    RagSearchRequest,
    RagSearchResponse,
    RagSearchResultItem,
    SLARequest,
    SLAResponse,
    SentimentRequest,
    SentimentResponse,
    StructuredEntity,
)
from .cloud_extractors import ExtractionError, extract_gstin_details, extract_kyc_entities
from .config import API_CONFIG, MODELS_DIR
from .features import (
    AccessControl,
    CodeMixedConversationalInterface,
    ContractAnalyzer,
    CorporateKYCNERExtractor,
    CryptographicAuditTrail,
    EnterpriseRAG,
    GSTINReconciliationAgent,
    InvoiceParser,
    InvoiceValidator,
    MeetingIntelligence,
    MerkleTrees,
    ONDCRouter,
    SentimentAnalyzer,
    SLABreachPredictor,
    SelfHealingEngine,
    StateCheckpointing,
    TaxPlanningAgent,
    TemporalTriggers,
    VendorPerformanceScorer,
    WorkflowObservability,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_START = time()

FEATURE_CATALOG = [
    {
        "key": "code_mixed",
        "title": "Code-Mixed Conversational Interface",
        "category": "Language & Speech",
        "summary": "Understand Hinglish and extract intent from bilingual queries.",
        "interactive": True,
    },
    {
        "key": "meeting_intelligence",
        "title": "Autonomous Meeting Intelligence",
        "category": "Language & Speech",
        "summary": "Process meeting audio and return summaries, speakers, and action items.",
        "interactive": True,
    },
    {
        "key": "sentiment_analyzer",
        "title": "Cross-Lingual Sentiment Analyzer",
        "category": "Language & Speech",
        "summary": "Fast text sentiment scoring for English and Indian-language inputs.",
        "interactive": True,
    },
    {
        "key": "kyc_ner",
        "title": "Corporate KYC NER Extractor",
        "category": "Document Processing",
        "summary": "Extract GSTIN, PAN, CIN, contacts, and company entities from KYC text.",
        "interactive": True,
    },
    {
        "key": "invoice_parser",
        "title": "Indic Vision-Language Invoice Parser",
        "category": "Document Processing",
        "summary": "Upload invoice imagery and extract structured invoice fields.",
        "interactive": True,
    },
    {
        "key": "contract_analyzer",
        "title": "Multilingual Contract Analyzer",
        "category": "Document Processing",
        "summary": "Frontend-ready module card for future legal review workflows.",
        "interactive": False,
    },
    {
        "key": "invoice_validator",
        "title": "Smart e-Invoice Predictive Validator",
        "category": "Document Processing",
        "summary": "Reserved validator surface for future invoice autofill and checks.",
        "interactive": False,
    },
    {
        "key": "sla_predictor",
        "title": "Agentic SLA Breach Predictor",
        "category": "AI & ML Models",
        "summary": "Predict breach probability and generate operator recommendations.",
        "interactive": True,
    },
    {
        "key": "vendor_scorer",
        "title": "Vendor Performance Scorer",
        "category": "AI & ML Models",
        "summary": "Placeholder feature card preserved for roadmap completeness.",
        "interactive": False,
    },
    {
        "key": "tax_planning",
        "title": "Tax-Context Action Planning Agent",
        "category": "AI & ML Models",
        "summary": "Reserved card for future tax planning workflows.",
        "interactive": False,
    },
    {
        "key": "cryptographic_audit",
        "title": "Zero-Cost Cryptographic Audit Trail",
        "category": "Security & Audit",
        "summary": "Log actions, verify the chain, and inspect tamper-proof events.",
        "interactive": True,
    },
    {
        "key": "access_control",
        "title": "Zero-Trust Autonomous Access Control",
        "category": "Security & Audit",
        "summary": "Placeholder security module card retained in the new UI.",
        "interactive": False,
    },
    {
        "key": "merkle_trees",
        "title": "Cryptographic Python Merkle Trees",
        "category": "Security & Audit",
        "summary": "Preview card for future integrity workflows and proofs.",
        "interactive": False,
    },
    {
        "key": "self_healing",
        "title": "Self-Healing Execution Engine",
        "category": "Workflow & Automation",
        "summary": "Reserved orchestration card to keep the suite complete.",
        "interactive": False,
    },
    {
        "key": "workflow_observability",
        "title": "Local Agentic Workflow Observability",
        "category": "Workflow & Automation",
        "summary": "Preview observability card for internal workflow telemetry.",
        "interactive": False,
    },
    {
        "key": "state_checkpointing",
        "title": "Local System State Checkpointing",
        "category": "Workflow & Automation",
        "summary": "Placeholder card for future save-and-resume execution flows.",
        "interactive": False,
    },
    {
        "key": "temporal_triggers",
        "title": "Temporal Logic Trigger",
        "category": "Workflow & Automation",
        "summary": "Reserved automation card for deadline-driven alerts and tasks.",
        "interactive": False,
    },
    {
        "key": "gstin_reconciliation",
        "title": "Autonomous GSTIN Reconciliation Agent",
        "category": "Integration & Routing",
        "summary": "Extract GSTINs, reconcile supplier data, and inspect fuzzy matches.",
        "interactive": True,
    },
    {
        "key": "ondc_router",
        "title": "Multi-Agent ONDC Router",
        "category": "Integration & Routing",
        "summary": "Preview card for natural language to ONDC routing workflows.",
        "interactive": False,
    },
    {
        "key": "enterprise_rag",
        "title": "Local Enterprise RAG Memory",
        "category": "Integration & Routing",
        "summary": "Add documents, search policies, and surface context-backed answers.",
        "interactive": True,
    },
]

KYC_PROVIDER_OPTIONS = [
    ("OpenAI Structured Outputs", ["OPENAI_API_KEY"]),
    ("Groq JSON Mode", ["GROQ_API_KEY"]),
]
INVOICE_PROVIDER_OPTIONS = [
    ("Sarvam Vision API", ["SARVAM_VISION_API_KEY"]),
    (
        "Google Document AI",
        [
            "GOOGLE_DOCUMENT_AI_PROJECT_ID",
            "GOOGLE_DOCUMENT_AI_LOCATION",
            "GOOGLE_DOCUMENT_AI_PROCESSOR_ID",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ],
    ),
]
MEETING_PROVIDER_OPTIONS = [
    ("Bhashini", ["BHASHINI_API_KEY"]),
    ("Sarvam Audio", ["SARVAM_AUDIO_API_KEY"]),
]


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def resolve_provider(options: List[tuple[str, List[str]]]) -> tuple[Optional[str], bool, List[str]]:
    required_env = sorted({item for _, envs in options for item in envs})

    for provider, envs in options:
        if all(os.getenv(env) for env in envs):
            return provider, True, required_env

    return None, False, required_env


def serialize_audit_integrity(result: Dict[str, Any], *, tool: str) -> AuditVerifyResponse:
    return AuditVerifyResponse(
        status="ok" if not result.get("error") else "error",
        tool=tool,
        mode="local_cryptographic_chain",
        message=result.get("error"),
        total_entries=int(result.get("total_entries", 0)),
        valid_entries=int(result.get("valid_entries", 0)),
        invalid_entries=int(result.get("invalid_entries", 0)),
        chain_broken=bool(result.get("chain_broken", False)),
        integrity_score=float(result.get("integrity_score", 0.0)),
        issues=list(result.get("issues", [])),
    )


def serialize_audit_entry(entry: Any) -> Optional[AuditEntryRecord]:
    if entry is None:
        return None

    entry_data = to_jsonable(entry)
    return AuditEntryRecord(**entry_data)


def serialize_kyc_preview(raw: Dict[str, Any]) -> tuple[List[StructuredEntity], Dict[str, Any]]:
    grouped_entities = raw.get("grouped_entities", {}) or {}
    entities: List[StructuredEntity] = []

    for label, items in grouped_entities.items():
        for item in items:
            entities.append(
                StructuredEntity(
                    label=str(label),
                    value=str(item.get("text", "")),
                    confidence=float(item.get("confidence", 0.0)),
                )
            )

    return entities, to_jsonable(raw.get("summary", {}))


class FeatureRegistry:
    """Lazy-load and cache backend features only when they are requested."""

    def __init__(self) -> None:
        self._instances: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._seeded_rag = False

    def get(self, key: str) -> Any:
        with self._lock:
            if key in self._instances:
                return self._instances[key]

            logger.info("Initializing feature '%s' for web request", key)
            instance = self._build_feature(key)
            self._instances[key] = instance
            return instance

    def loaded_features(self) -> List[str]:
        return sorted(self._instances.keys())

    def _build_feature(self, key: str) -> Any:
        if key == "code_mixed":
            return CodeMixedConversationalInterface()
        if key == "sentiment_analyzer":
            return SentimentAnalyzer()
        if key == "gstin_reconciliation":
            return GSTINReconciliationAgent()
        if key == "kyc_ner":
            return CorporateKYCNERExtractor()
        if key == "sla_predictor":
            return self._build_sla_predictor()
        if key == "enterprise_rag":
            rag = EnterpriseRAG()
            self._seed_rag_library(rag)
            return rag
        if key == "cryptographic_audit":
            return CryptographicAuditTrail()
        if key == "invoice_parser":
            return InvoiceParser()
        if key == "meeting_intelligence":
            return MeetingIntelligence()
        if key == "contract_analyzer":
            return ContractAnalyzer()
        if key == "workflow_observability":
            return WorkflowObservability()
        if key == "invoice_validator":
            return InvoiceValidator()
        if key == "access_control":
            return AccessControl()
        if key == "vendor_scorer":
            return VendorPerformanceScorer()
        if key == "state_checkpointing":
            return StateCheckpointing()
        if key == "tax_planning":
            return TaxPlanningAgent()
        if key == "self_healing":
            return SelfHealingEngine()
        if key == "temporal_triggers":
            return TemporalTriggers()
        if key == "merkle_trees":
            return MerkleTrees()
        if key == "ondc_router":
            return ONDCRouter()
        raise KeyError(f"Unknown feature key: {key}")

    def _build_sla_predictor(self) -> SLABreachPredictor:
        predictor = SLABreachPredictor()
        model_path = MODELS_DIR / "sla_predictor.joblib"

        try:
            if model_path.exists():
                predictor.load_model(str(model_path))
            else:
                training_frame = predictor.generate_synthetic_data(num_samples=900)
                predictor.train_model(training_frame)
                predictor.save_model(str(model_path))
        except Exception as exc:
            logger.warning("SLA predictor model warm-up failed, retraining inline: %s", exc)
            training_frame = predictor.generate_synthetic_data(num_samples=600)
            predictor.train_model(training_frame)

        return predictor

    def _seed_rag_library(self, rag: EnterpriseRAG) -> None:
        if self._seeded_rag or rag.documents:
            return

        sample_docs = [
            {
                "title": "Enterprise Incident Response Policy",
                "category": "Policy",
                "content": (
                    "Critical incidents must be acknowledged within 15 minutes, triaged by a senior "
                    "operator, and escalated to leadership if business services are impacted for more "
                    "than 30 minutes."
                ),
                "author": "Security Office",
                "tags": ["incident", "sla", "security"],
            },
            {
                "title": "Vendor GST Verification Playbook",
                "category": "Procedure",
                "content": (
                    "Finance teams must validate GSTIN format, compare supplier legal names against "
                    "the registry, and flag address mismatches above the approved reconciliation threshold."
                ),
                "author": "Finance Ops",
                "tags": ["gstin", "vendor", "finance"],
            },
            {
                "title": "KYC Intake Checklist",
                "category": "Compliance",
                "content": (
                    "Every KYC packet should include company name, PAN, GSTIN, CIN, registered address, "
                    "authorized signatory details, and verifiable contact information before approval."
                ),
                "author": "Compliance Desk",
                "tags": ["kyc", "compliance", "identity"],
            },
        ]

        for doc in sample_docs:
            rag.add_document(
                title=doc["title"],
                content=doc["content"],
                category=doc["category"],
                author=doc["author"],
                tags=doc["tags"],
            )

        self._seeded_rag = True


registry = FeatureRegistry()

app = FastAPI(
    title="Enterprise Control Plane API",
    description="Typed FastAPI backend for the Next.js enterprise automation platform.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_ORIGIN", "http://localhost:3000"),
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "Enterprise Control Plane API",
        "status": "ok",
        "docs": "/docs",
        "api_root": "/api",
    }


@app.get("/api/health", response_model=PlatformHealthResponse)
def health() -> PlatformHealthResponse:
    return PlatformHealthResponse(
        status="ok",
        ui="nextjs-fastapi-platform",
        streamlit_replaced=True,
        uptime_seconds=round(time() - APP_START, 2),
        python_version=sys.version.split()[0],
        api_port=API_CONFIG["port"],
        feature_count=len(FEATURE_CATALOG),
        interactive_count=sum(1 for feature in FEATURE_CATALOG if feature["interactive"]),
        loaded_features=registry.loaded_features(),
    )


@app.get("/api/features", response_model=FeatureCatalogResponse)
def features() -> FeatureCatalogResponse:
    loaded = set(registry.loaded_features())
    items = [
        FeatureCard(**feature, loaded=feature["key"] in loaded) for feature in FEATURE_CATALOG
    ]

    return FeatureCatalogResponse(
        features=items,
        categories=sorted({feature["category"] for feature in FEATURE_CATALOG}),
    )


@app.post("/api/analyze/code-mixed", response_model=CodeMixedResponse)
def analyze_code_mixed(payload: CodeMixedRequest) -> CodeMixedResponse:
    feature = registry.get("code_mixed")
    result = to_jsonable(feature.understand_code_mixed_text(payload.text))
    entities = [
        DetectedEntity(
            type=str(item.get("type", "unknown")),
            value=str(item.get("value", "")),
            start=item.get("start"),
            end=item.get("end"),
        )
        for item in result.get("entities", [])
    ]

    return CodeMixedResponse(
        status="ok",
        tool="code_mixed",
        mode="local_transformers_fallback",
        message="Intent analysis completed locally.",
        input_text=payload.text,
        primary_language=str(result.get("language_info", {}).get("primary_lang", "unknown")),
        intent=str(result.get("intent", "general")),
        confidence=float(result.get("confidence", 0.0)),
        response_text=str(result.get("response", "")),
        entities=entities,
        raw=result,
    )


@app.post("/api/analyze/sentiment", response_model=SentimentResponse)
def analyze_sentiment(payload: SentimentRequest) -> SentimentResponse:
    feature = registry.get("sentiment_analyzer")
    result = to_jsonable(feature.analyze_sentiment(payload.text, payload.threshold))

    return SentimentResponse(
        status="ok",
        tool="sentiment_analyzer",
        mode=str(result.get("mode", "local")),
        message="Sentiment scoring completed locally.",
        label=str(result.get("sentiment", "neutral")),
        confidence=float(result.get("confidence", 0.0)),
        threshold=float(result.get("threshold", payload.threshold)),
        above_threshold=bool(result.get("above_threshold", False)),
        language=str(result.get("language", "auto")),
        model=str(result.get("model", "unknown")),
        distribution={
            str(key): float(value) for key, value in result.get("distribution", {}).items()
        },
    )


@app.post("/api/extract/gstin", response_model=GSTINResponse)
def extract_gstin(payload: GSTINRequest) -> GSTINResponse:
    feature = registry.get("gstin_reconciliation")
    provider, configured, required_env = resolve_provider(KYC_PROVIDER_OPTIONS)

    if configured:
        try:
            active_provider, extraction = extract_gstin_details(payload.text, payload.company_name)
            return GSTINResponse(
                status="ok",
                tool="gstin_reconciliation",
                mode="cloud_structured_output",
                configured=True,
                provider=active_provider,
                message="GSTIN extraction completed with structured cloud output.",
                gstins=extraction.gstins,
                company_name=extraction.company_name or payload.company_name,
                reconciliation=to_jsonable(extraction.reconciliation),
                raw=to_jsonable(extraction),
            )
        except ExtractionError as exc:
            logger.warning("Structured GSTIN extraction failed, using local preview: %s", exc)

    extraction = to_jsonable(feature.intelligent_gstin_extraction(payload.text))
    company_name = payload.company_name
    if not company_name:
        companies = extraction.get("extracted_data", {}).get("company_names", [])
        company_name = companies[0] if companies else None

    reconciliation = to_jsonable(
        feature.reconcile_supplier_data(
            {"company_name": company_name or "", "description": payload.text}
        )
    )

    return GSTINResponse(
        status="ok",
        tool="gstin_reconciliation",
        mode="local_preview_fallback",
        configured=configured,
        provider=provider,
        message=(
            "Returning local preview extraction. Add OPENAI_API_KEY or GROQ_API_KEY for "
            "strict cloud structured outputs."
        ),
        required_env=[] if configured else required_env,
        gstins=list(extraction.get("validated_gstins", [])),
        company_name=company_name,
        reconciliation=reconciliation,
        raw={"extraction": extraction, "reconciliation": reconciliation},
    )


@app.post("/api/extract/kyc", response_model=KYCResponse)
def extract_kyc(payload: CodeMixedRequest) -> KYCResponse:
    feature = registry.get("kyc_ner")
    provider, configured, required_env = resolve_provider(KYC_PROVIDER_OPTIONS)

    if configured:
        try:
            active_provider, extraction = extract_kyc_entities(payload.text)
            return KYCResponse(
                status="ok",
                tool="kyc_ner",
                mode="cloud_structured_output",
                configured=True,
                provider=active_provider,
                message="KYC entities extracted with schema-validated cloud output.",
                entities=[
                    StructuredEntity(
                        label=item.label,
                        value=item.value,
                        confidence=item.confidence,
                    )
                    for item in extraction.entities
                ],
                document_summary=to_jsonable(extraction.summary),
                raw=to_jsonable(extraction),
            )
        except ExtractionError as exc:
            logger.warning("Structured KYC extraction failed, using local preview: %s", exc)

    raw = to_jsonable(feature.process_document(payload.text))
    entities, summary = serialize_kyc_preview(raw)

    return KYCResponse(
        status="ok",
        tool="kyc_ner",
        mode="local_preview_fallback",
        configured=configured,
        provider=provider,
        message=(
            "Returning local preview extraction. Add OPENAI_API_KEY or GROQ_API_KEY for "
            "strict cloud structured outputs."
        ),
        required_env=[] if configured else required_env,
        entities=entities,
        document_summary=summary,
        raw=raw,
    )


@app.post("/api/predict/sla", response_model=SLAResponse)
def predict_sla(payload: SLARequest) -> SLAResponse:
    feature = registry.get("sla_predictor")
    result = to_jsonable(feature.predict_breach_risk(payload.model_dump()))

    return SLAResponse(
        status="ok",
        tool="sla_predictor",
        mode="local_xgboost_or_random_forest",
        message="SLA breach risk scored using the local model.",
        ticket_id=str(result.get("ticket_id", payload.ticket_id or "WEB-TICKET")),
        breach_probability=float(result.get("breach_probability", 0.0)),
        risk_level=str(result.get("risk_level", "Unknown")),
        confidence=float(result.get("confidence", 0.0)),
        breach_prediction=int(result.get("breach_prediction", 0)),
        recommendations=[str(item) for item in result.get("recommendations", [])],
        feature_contributions=to_jsonable(result.get("feature_contributions", {})),
    )


@app.post("/api/rag/documents", response_model=RagDocumentResponse)
def add_rag_document(payload: RagDocumentRequest) -> RagDocumentResponse:
    feature = registry.get("enterprise_rag")
    document_id = feature.add_document(
        title=payload.title,
        content=payload.content,
        category=payload.category,
        author=payload.author,
        tags=payload.tags,
    )
    statistics = to_jsonable(feature.get_statistics())

    return RagDocumentResponse(
        status="ok",
        tool="enterprise_rag",
        mode="local_vector_memory",
        message="Document added to enterprise memory.",
        document_id=document_id,
        total_documents=int(statistics.get("total_documents", 0)),
        categories=feature.get_categories(),
        statistics=statistics,
    )


@app.post("/api/rag/search", response_model=RagSearchResponse)
def search_rag(payload: RagSearchRequest) -> RagSearchResponse:
    feature = registry.get("enterprise_rag")
    results = feature.search(payload.query, top_k=payload.top_k, category=payload.category)

    return RagSearchResponse(
        status="ok",
        tool="enterprise_rag",
        mode="local_vector_memory",
        message="Enterprise memory query completed.",
        results=[
            RagSearchResultItem(
                title=result.document.title,
                category=result.document.category,
                author=result.document.author,
                similarity_score=float(result.similarity_score),
                answer=str(result.answer),
                relevant_chunks=[str(chunk) for chunk in result.relevant_chunks],
            )
            for result in results
        ],
        categories=feature.get_categories(),
        statistics=to_jsonable(feature.get_statistics()),
    )


@app.post("/api/audit/log", response_model=AuditLogResponse)
def audit_log(payload: AuditLogRequest) -> AuditLogResponse:
    feature = registry.get("cryptographic_audit")
    entry_id = feature.log_action(payload.action, payload.user_id, payload.resource, payload.details)
    entry = feature.get_entry(entry_id)
    integrity_raw = feature.verify_chain_integrity()

    return AuditLogResponse(
        status="ok",
        tool="cryptographic_audit",
        mode="local_cryptographic_chain",
        message="Audit entry appended and verified locally.",
        entry_id=entry_id,
        entry=serialize_audit_entry(entry),
        integrity=serialize_audit_integrity(integrity_raw, tool="cryptographic_audit"),
    )


@app.get("/api/audit/verify", response_model=AuditVerifyResponse)
def audit_verify() -> AuditVerifyResponse:
    feature = registry.get("cryptographic_audit")
    integrity_raw = feature.verify_chain_integrity()
    return serialize_audit_integrity(integrity_raw, tool="cryptographic_audit")


@app.post("/api/invoice/parse", response_model=InvoiceParseResponse)
def parse_invoice(file: UploadFile = File(...)) -> InvoiceParseResponse:
    feature = registry.get("invoice_parser")
    provider, configured, required_env = resolve_provider(INVOICE_PROVIDER_OPTIONS)
    suffix = Path(file.filename or "invoice").suffix or ".png"
    temp_path: Optional[Path] = None

    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file.file.read())
            temp_path = Path(temp_file.name)

        invoice_data = to_jsonable(feature.parse_invoice(str(temp_path)))
        validation = to_jsonable(feature.validate_extracted_data(invoice_data))

        return InvoiceParseResponse(
            status="ok",
            tool="invoice_parser",
            mode="local_preview_fallback",
            configured=configured,
            provider=provider,
            message=(
                "Local invoice parser returned a preview result. Add Sarvam Vision or Google "
                "Document AI credentials for production OCR coverage."
            ),
            required_env=[] if configured else required_env,
            file_name=file.filename,
            invoice=invoice_data,
            validation=validation,
            raw={"invoice": invoice_data, "validation": validation},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Invoice parsing failed: {exc}") from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.post("/api/meeting/analyze", response_model=MeetingAnalysisResponse)
def analyze_meeting(file: UploadFile = File(...)) -> MeetingAnalysisResponse:
    provider, configured, required_env = resolve_provider(MEETING_PROVIDER_OPTIONS)

    return MeetingAnalysisResponse(
        status="needs_configuration",
        tool="meeting_intelligence",
        mode="cloud_transcription_placeholder",
        configured=configured,
        provider=provider,
        message=(
            "Meeting analysis is intentionally blocked until a real audio provider is wired. "
            "Add BHASHINI_API_KEY or SARVAM_AUDIO_API_KEY and connect the provider client to "
            "avoid fabricated transcripts."
        ),
        required_env=[] if configured else required_env,
        file_name=file.filename,
        summary={},
        report=None,
        raw={},
    )
